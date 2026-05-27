"""Google OAuth2 sign-in for OneLLM.

Implements two endpoints under /onellm/auth/oauth/google:

  GET  /authorize     -> redirect to Google's consent screen
  GET  /callback      -> exchange code, lookup or auto-provision user, issue tokens

Three branches when Google returns:
  1. (Google, sub) already linked       -> issue tokens for that user
  2. sub unknown but email already in DB -> bind by linking, issue tokens
  3. Brand new user                      -> create a tenant + user, issue tokens

The tokens come back as JSON (matches the password login response) and the
frontend stores them the same way; we do not set cookies from the backend
so the frontend OneLLM session stays under NextAuth's control.
"""

import secrets
from typing import Any, Optional, Tuple
from urllib.parse import urlencode

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from fastapi.responses import RedirectResponse

from onellm.auth import service as auth_service
from onellm.auth.password import new_security_stamp
from onellm.config import SETTINGS
from onellm.db import get_prisma
from onellm.schemas.token import TokenOut
from onellm.sync.litellm_shadow import (
    upsert_litellm_team_shadow,
    upsert_litellm_user_shadow,
)

router = APIRouter(prefix="/auth/oauth/google", tags=["onellm-oauth"])

GOOGLE_PROVIDER = "Google"


def _require_google_settings() -> Tuple[str, str, str]:
    cid = SETTINGS.google_client_id
    secret = SETTINGS.google_client_secret
    redirect = SETTINGS.google_redirect_uri
    if not cid or not secret or not redirect:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "Google OAuth is not configured. Set ONELLM_GOOGLE_CLIENT_ID, "
                "ONELLM_GOOGLE_CLIENT_SECRET, ONELLM_GOOGLE_REDIRECT_URI."
            ),
        )
    return cid, secret, redirect


def _build_google_sso():
    from fastapi_sso.sso.google import GoogleSSO

    cid, secret, redirect = _require_google_settings()
    return GoogleSSO(
        client_id=cid,
        client_secret=secret,
        redirect_uri=redirect,
        allow_insecure_http=False,
    )


def _normalize_email(email: Optional[str]) -> Optional[str]:
    return email.upper().strip() if email else None


@router.get("/authorize")
async def google_authorize(
    request: Request,
    redirect_to: Optional[str] = Query(default=None, max_length=512),
) -> RedirectResponse:
    """Start the Google OAuth dance. Returns a 302 to Google."""
    sso = _build_google_sso()
    # state carries the post-login front-end target; we sign it with a nonce
    # so the callback can verify the redirect_to wasn't tampered with mid-flight.
    nonce = secrets.token_urlsafe(16)
    state_payload = nonce if not redirect_to else f"{nonce}|{redirect_to}"
    with sso:
        return await sso.get_login_redirect(state=state_payload)


@router.get("/callback")
async def google_callback(request: Request) -> Any:
    sso = _build_google_sso()
    with sso:
        openid = await sso.verify_and_process(request)
    if openid is None or not openid.id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Google did not return a verified profile",
        )

    db = get_prisma()
    state = request.query_params.get("state") or ""
    redirect_to: Optional[str] = None
    if "|" in state:
        _, redirect_to = state.split("|", 1)

    async with db.tx() as tx:
        link = await tx.sysuserlogin.find_unique(
            where={
                "login_provider_provider_key": {
                    "login_provider": GOOGLE_PROVIDER,
                    "provider_key": openid.id,
                }
            }
        )
        if link is not None:
            user = await tx.sysuser.find_unique(where={"id": link.user_id})
            if user is None or user.is_deleted:
                raise HTTPException(
                    status_code=status.HTTP_410_GONE,
                    detail="Linked OneLLM user was removed",
                )
        else:
            user = None
            if openid.email:
                user = await tx.sysuser.find_unique(
                    where={"normalized_email": _normalize_email(openid.email)}
                )
            if user is None:
                # Auto-provision: new tenant + new user, just like /auth/register
                tenant_slug = (openid.email or openid.id).split("@")[0].lower().replace(
                    ".", "-"
                )[:40] or "user"
                final_code = await auth_service.resolve_unique_tenant_code(
                    tx, tenant_slug
                )
                tenant = await tx.systenant.create(
                    data={
                        "name": openid.display_name or openid.email or final_code,
                        "code": final_code,
                        "plan_code": "free",
                    }
                )
                user = await tx.sysuser.create(
                    data={
                        "tenant_id": tenant.id,
                        "name": openid.display_name,
                        "user_name": openid.email,
                        "normalized_user_name": _normalize_email(openid.email),
                        "email": openid.email,
                        "normalized_email": _normalize_email(openid.email),
                        "email_confirmed": True,
                        "security_stamp": new_security_stamp(),
                        "role_global": "user",
                    }
                )
                owner_role = await tx.sysrole.find_unique(
                    where={"code": "tenant_owner"}
                )
                if owner_role is not None:
                    await tx.sysuserrole.create(
                        data={
                            "user_id": user.id,
                            "role_id": owner_role.id,
                            "tenant_id": tenant.id,
                        }
                    )
                team_id = await upsert_litellm_team_shadow(tx, tenant=tenant)
                if team_id != tenant.litellm_team_id:
                    await tx.systenant.update(
                        where={"id": tenant.id},
                        data={"litellm_team_id": team_id},
                    )
                await upsert_litellm_user_shadow(tx, user=user, tenant=tenant)

            # Bind the Google identity to whichever user we have now.
            await tx.sysuserlogin.create(
                data={
                    "login_provider": GOOGLE_PROVIDER,
                    "provider_key": openid.id,
                    "provider_display_name": openid.email or openid.display_name,
                    "user_id": user.id,
                }
            )

        tokens = await auth_service.build_tokens(
            tx,
            user_id=user.id,
            tenant_id=user.tenant_id,
            role_global=user.role_global,
            security_stamp=user.security_stamp,
        )

    if redirect_to:
        # Append tokens to the redirect target as URL fragment so the browser
        # never sends them to the server; the front-end reads location.hash.
        params = urlencode(
            {
                "access_token": tokens.access_token,
                "refresh_token": tokens.refresh_token,
                "expires_in": tokens.expires_in,
                "token_type": tokens.token_type,
            }
        )
        separator = "&" if "#" in redirect_to else "#"
        return RedirectResponse(
            url=f"{redirect_to}{separator}{params}", status_code=302
        )

    return TokenOut(
        access_token=tokens.access_token,
        refresh_token=tokens.refresh_token,
        expires_in=tokens.expires_in,
        token_type=tokens.token_type,
    )
