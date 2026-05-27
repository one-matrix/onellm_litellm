-- OneLLM identity / multi-tenant / RBAC tables.
--
-- Ported from docs/sql/user/public.sql (ASP.NET Identity style) into the
-- `app` schema so upstream litellm migrations never collide. Three OneLLM
-- adaptations vs the original SQL:
--   * sys_user_roles is tenant-scoped: PK extended to (user_id, role_id, tenant_id)
--   * sys_tenants adds litellm_team_id / plan_code / default_price_markup
--   * sys_users adds litellm_user_id / role_global ("root" | "admin" | "user")
--   * the original fk_sys_users_org_departments_department_id is dropped — we
--     keep the `department_id` column for future use but do NOT FK into a
--     table that does not belong to OneLLM.
--
-- Idempotent: safe to re-run on a partially-applied DB.

CREATE SCHEMA IF NOT EXISTS "app";

-- 1. sys_tenants -------------------------------------------------------------
CREATE TABLE IF NOT EXISTS "app"."sys_tenants" (
    "id"                    UUID NOT NULL DEFAULT gen_random_uuid(),
    "name"                  VARCHAR(100) NOT NULL,
    "code"                  VARCHAR(50)  NOT NULL,
    "domain"                VARCHAR(200),
    "is_active"             BOOLEAN      NOT NULL DEFAULT TRUE,

    -- OneLLM extensions
    "litellm_team_id"       TEXT,
    "plan_code"             VARCHAR(50)  NOT NULL DEFAULT 'free',
    "default_price_markup"  DOUBLE PRECISION NOT NULL DEFAULT 1.0,

    "created_at"            TIMESTAMPTZ(6) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at"            TIMESTAMPTZ(6),
    "is_deleted"            BOOLEAN      NOT NULL DEFAULT FALSE,

    CONSTRAINT "sys_tenants_pkey" PRIMARY KEY ("id")
);

CREATE UNIQUE INDEX IF NOT EXISTS "sys_tenants_code_key"
    ON "app"."sys_tenants"("code");
CREATE INDEX IF NOT EXISTS "sys_tenants_litellm_team_id_idx"
    ON "app"."sys_tenants"("litellm_team_id");

-- 2. sys_users ---------------------------------------------------------------
CREATE TABLE IF NOT EXISTS "app"."sys_users" (
    "id"                     UUID NOT NULL DEFAULT gen_random_uuid(),
    "tenant_id"              UUID,                                                -- primary/home tenant
    "department_id"          UUID,                                                -- reserved, no FK
    "name"                   VARCHAR(100),
    "is_active"              BOOLEAN NOT NULL DEFAULT TRUE,
    "created_at"             TIMESTAMPTZ(6) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at"             TIMESTAMPTZ(6) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "is_deleted"             BOOLEAN NOT NULL DEFAULT FALSE,

    -- ASP.NET Identity baseline
    "user_name"              VARCHAR(256),
    "normalized_user_name"   VARCHAR(256),
    "email"                  VARCHAR(256),
    "normalized_email"       VARCHAR(256),
    "email_confirmed"        BOOLEAN NOT NULL DEFAULT FALSE,
    "password_hash"          TEXT,
    "security_stamp"         TEXT,
    "concurrency_stamp"      TEXT,
    "phone_number"           TEXT,
    "phone_number_confirmed" BOOLEAN NOT NULL DEFAULT FALSE,
    "two_factor_enabled"     BOOLEAN NOT NULL DEFAULT FALSE,
    "lockout_end"            TIMESTAMPTZ(6),
    "lockout_enabled"        BOOLEAN NOT NULL DEFAULT TRUE,
    "access_failed_count"    INTEGER NOT NULL DEFAULT 0,

    -- OneLLM extensions
    "litellm_user_id"        TEXT,
    "role_global"            VARCHAR(20) NOT NULL DEFAULT 'user',

    CONSTRAINT "sys_users_pkey" PRIMARY KEY ("id"),
    CONSTRAINT "sys_users_tenant_fkey"
        FOREIGN KEY ("tenant_id") REFERENCES "app"."sys_tenants"("id")
        ON DELETE SET NULL ON UPDATE NO ACTION,
    CONSTRAINT "chk_sys_users_role_global"
        CHECK ("role_global" IN ('root', 'admin', 'user'))
);

CREATE UNIQUE INDEX IF NOT EXISTS "sys_users_normalized_user_name_key"
    ON "app"."sys_users"("normalized_user_name");
CREATE UNIQUE INDEX IF NOT EXISTS "sys_users_normalized_email_key"
    ON "app"."sys_users"("normalized_email");
CREATE INDEX IF NOT EXISTS "sys_users_tenant_id_idx"
    ON "app"."sys_users"("tenant_id");
CREATE INDEX IF NOT EXISTS "sys_users_litellm_user_id_idx"
    ON "app"."sys_users"("litellm_user_id");

-- 3. sys_roles ---------------------------------------------------------------
CREATE TABLE IF NOT EXISTS "app"."sys_roles" (
    "id"                UUID NOT NULL DEFAULT gen_random_uuid(),
    "code"              VARCHAR(50)  NOT NULL,
    "name"              VARCHAR(256),
    "normalized_name"   VARCHAR(256),
    "description"       VARCHAR(200),
    "concurrency_stamp" TEXT,
    "created_at"        TIMESTAMPTZ(6) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at"        TIMESTAMPTZ(6),
    "is_deleted"        BOOLEAN NOT NULL DEFAULT FALSE,

    CONSTRAINT "sys_roles_pkey" PRIMARY KEY ("id")
);

CREATE UNIQUE INDEX IF NOT EXISTS "sys_roles_code_key"
    ON "app"."sys_roles"("code");
CREATE UNIQUE INDEX IF NOT EXISTS "sys_roles_normalized_name_key"
    ON "app"."sys_roles"("normalized_name");

-- 4. sys_user_roles (tenant-scoped) ------------------------------------------
CREATE TABLE IF NOT EXISTS "app"."sys_user_roles" (
    "user_id"   UUID NOT NULL,
    "role_id"   UUID NOT NULL,
    "tenant_id" UUID NOT NULL,
    "created_at" TIMESTAMPTZ(6) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "sys_user_roles_pkey" PRIMARY KEY ("user_id", "role_id", "tenant_id"),
    CONSTRAINT "sys_user_roles_user_fkey"
        FOREIGN KEY ("user_id") REFERENCES "app"."sys_users"("id")
        ON DELETE CASCADE ON UPDATE NO ACTION,
    CONSTRAINT "sys_user_roles_role_fkey"
        FOREIGN KEY ("role_id") REFERENCES "app"."sys_roles"("id")
        ON DELETE CASCADE ON UPDATE NO ACTION,
    CONSTRAINT "sys_user_roles_tenant_fkey"
        FOREIGN KEY ("tenant_id") REFERENCES "app"."sys_tenants"("id")
        ON DELETE CASCADE ON UPDATE NO ACTION
);

CREATE INDEX IF NOT EXISTS "sys_user_roles_role_id_idx"
    ON "app"."sys_user_roles"("role_id");
CREATE INDEX IF NOT EXISTS "sys_user_roles_tenant_id_idx"
    ON "app"."sys_user_roles"("tenant_id");

-- 5. sys_permissions ---------------------------------------------------------
CREATE TABLE IF NOT EXISTS "app"."sys_permissions" (
    "id"          UUID NOT NULL DEFAULT gen_random_uuid(),
    "name"        VARCHAR(50)  NOT NULL,
    "code"        VARCHAR(100) NOT NULL,
    "type"        TEXT         NOT NULL,
    "path"        VARCHAR(200),
    "icon"        VARCHAR(50),
    "sort_order"  INTEGER      NOT NULL DEFAULT 0,
    "parent_id"   UUID,
    "created_at"  TIMESTAMPTZ(6) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at"  TIMESTAMPTZ(6),
    "is_deleted"  BOOLEAN      NOT NULL DEFAULT FALSE,

    CONSTRAINT "sys_permissions_pkey" PRIMARY KEY ("id"),
    CONSTRAINT "sys_permissions_parent_fkey"
        FOREIGN KEY ("parent_id") REFERENCES "app"."sys_permissions"("id")
        ON DELETE RESTRICT ON UPDATE NO ACTION
);

CREATE UNIQUE INDEX IF NOT EXISTS "sys_permissions_code_key"
    ON "app"."sys_permissions"("code");
CREATE INDEX IF NOT EXISTS "sys_permissions_parent_id_idx"
    ON "app"."sys_permissions"("parent_id");

-- 6. sys_role_permissions ----------------------------------------------------
CREATE TABLE IF NOT EXISTS "app"."sys_role_permissions" (
    "role_id"       UUID NOT NULL,
    "permission_id" UUID NOT NULL,

    CONSTRAINT "sys_role_permissions_pkey" PRIMARY KEY ("role_id", "permission_id"),
    CONSTRAINT "sys_role_permissions_role_fkey"
        FOREIGN KEY ("role_id") REFERENCES "app"."sys_roles"("id")
        ON DELETE CASCADE ON UPDATE NO ACTION,
    CONSTRAINT "sys_role_permissions_permission_fkey"
        FOREIGN KEY ("permission_id") REFERENCES "app"."sys_permissions"("id")
        ON DELETE CASCADE ON UPDATE NO ACTION
);

CREATE INDEX IF NOT EXISTS "sys_role_permissions_permission_id_idx"
    ON "app"."sys_role_permissions"("permission_id");

-- 7. sys_user_logins (OAuth bindings) ----------------------------------------
CREATE TABLE IF NOT EXISTS "app"."sys_user_logins" (
    "login_provider"        TEXT NOT NULL,
    "provider_key"          TEXT NOT NULL,
    "provider_display_name" TEXT,
    "user_id"               UUID NOT NULL,
    "created_at"            TIMESTAMPTZ(6) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "sys_user_logins_pkey" PRIMARY KEY ("login_provider", "provider_key"),
    CONSTRAINT "sys_user_logins_user_fkey"
        FOREIGN KEY ("user_id") REFERENCES "app"."sys_users"("id")
        ON DELETE CASCADE ON UPDATE NO ACTION
);

CREATE INDEX IF NOT EXISTS "sys_user_logins_user_id_idx"
    ON "app"."sys_user_logins"("user_id");

-- 8. sys_user_tokens (refresh tokens, external tokens) -----------------------
CREATE TABLE IF NOT EXISTS "app"."sys_user_tokens" (
    "user_id"        UUID NOT NULL,
    "login_provider" TEXT NOT NULL,
    "name"           TEXT NOT NULL,
    "value"          TEXT,
    "expires_at"     TIMESTAMPTZ(6),
    "created_at"     TIMESTAMPTZ(6) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "sys_user_tokens_pkey" PRIMARY KEY ("user_id", "login_provider", "name"),
    CONSTRAINT "sys_user_tokens_user_fkey"
        FOREIGN KEY ("user_id") REFERENCES "app"."sys_users"("id")
        ON DELETE CASCADE ON UPDATE NO ACTION
);

CREATE INDEX IF NOT EXISTS "sys_user_tokens_expires_at_idx"
    ON "app"."sys_user_tokens"("expires_at");

-- 9. sys_user_claims ---------------------------------------------------------
CREATE TABLE IF NOT EXISTS "app"."sys_user_claims" (
    "id"          SERIAL,
    "user_id"     UUID NOT NULL,
    "claim_type"  TEXT,
    "claim_value" TEXT,

    CONSTRAINT "sys_user_claims_pkey" PRIMARY KEY ("id"),
    CONSTRAINT "sys_user_claims_user_fkey"
        FOREIGN KEY ("user_id") REFERENCES "app"."sys_users"("id")
        ON DELETE CASCADE ON UPDATE NO ACTION
);

CREATE INDEX IF NOT EXISTS "sys_user_claims_user_id_idx"
    ON "app"."sys_user_claims"("user_id");

-- 10. sys_role_claims --------------------------------------------------------
CREATE TABLE IF NOT EXISTS "app"."sys_role_claims" (
    "id"          SERIAL,
    "role_id"     UUID NOT NULL,
    "claim_type"  TEXT,
    "claim_value" TEXT,

    CONSTRAINT "sys_role_claims_pkey" PRIMARY KEY ("id"),
    CONSTRAINT "sys_role_claims_role_fkey"
        FOREIGN KEY ("role_id") REFERENCES "app"."sys_roles"("id")
        ON DELETE CASCADE ON UPDATE NO ACTION
);

CREATE INDEX IF NOT EXISTS "sys_role_claims_role_id_idx"
    ON "app"."sys_role_claims"("role_id");

-- ---------------------------------------------------------------------------
-- Seed: default roles (idempotent via ON CONFLICT (code))
-- ---------------------------------------------------------------------------
INSERT INTO "app"."sys_roles" ("id", "code", "name", "normalized_name", "description") VALUES
  (gen_random_uuid(), 'root',         'Root',         'ROOT',         'Platform super admin; global rights'),
  (gen_random_uuid(), 'tenant_owner', 'Tenant Owner', 'TENANT_OWNER', 'Tenant owner; full rights inside tenant'),
  (gen_random_uuid(), 'tenant_admin', 'Tenant Admin', 'TENANT_ADMIN', 'Manage members, keys, channels inside tenant'),
  (gen_random_uuid(), 'developer',    'Developer',    'DEVELOPER',    'Create keys + call models'),
  (gen_random_uuid(), 'billing',      'Billing',      'BILLING',      'View wallet + recharge'),
  (gen_random_uuid(), 'viewer',       'Viewer',       'VIEWER',       'Read-only visibility')
ON CONFLICT ("code") DO NOTHING;

-- ---------------------------------------------------------------------------
-- Seed: default permission tree (idempotent via ON CONFLICT (code))
-- type = 'group' (folder) | 'action' (leaf)
-- ---------------------------------------------------------------------------
INSERT INTO "app"."sys_permissions" ("id", "name", "code", "type", "sort_order") VALUES
  (gen_random_uuid(), 'Tenant',  'tenant',  'group', 10),
  (gen_random_uuid(), 'User',    'user',    'group', 20),
  (gen_random_uuid(), 'Key',     'key',     'group', 30),
  (gen_random_uuid(), 'Wallet',  'wallet',  'group', 40),
  (gen_random_uuid(), 'Model',   'model',   'group', 50),
  (gen_random_uuid(), 'Channel', 'channel', 'group', 60),
  (gen_random_uuid(), 'Task',    'task',    'group', 70)
ON CONFLICT ("code") DO NOTHING;

-- Leaf permissions; parent_id resolved via a CTE so we don't have to track UUIDs.
WITH parents AS (
    SELECT "id", "code" FROM "app"."sys_permissions" WHERE "type" = 'group'
)
INSERT INTO "app"."sys_permissions" ("id", "name", "code", "type", "parent_id", "sort_order")
SELECT gen_random_uuid(), data.name, data.code, 'action', parents."id", data.sort_order
FROM (
    VALUES
      ('Read tenant',   'tenant.read',    'tenant',  10),
      ('Update tenant', 'tenant.update',  'tenant',  20),
      ('Delete tenant', 'tenant.delete',  'tenant',  30),

      ('List members',  'user.read',      'user',    10),
      ('Invite member', 'user.invite',    'user',    20),
      ('Update member', 'user.update',    'user',    30),
      ('Remove member', 'user.delete',    'user',    40),

      ('List keys',     'key.read',       'key',     10),
      ('Create key',    'key.create',     'key',     20),
      ('Update key',    'key.update',     'key',     30),
      ('Delete key',    'key.delete',     'key',     40),

      ('View wallet',   'wallet.read',    'wallet',  10),
      ('Recharge',      'wallet.recharge','wallet',  20),
      ('View txs',      'wallet.tx.read', 'wallet',  30),

      ('List models',   'model.read',     'model',   10),
      ('Edit models',   'model.update',   'model',   20),

      ('List channels', 'channel.read',   'channel', 10),
      ('Edit channels', 'channel.update', 'channel', 20),

      ('List tasks',    'task.read',      'task',    10),
      ('Cancel task',   'task.cancel',    'task',    20)
) AS data(name, code, parent_code, sort_order)
JOIN parents ON parents."code" = data.parent_code
ON CONFLICT ("code") DO NOTHING;

-- ---------------------------------------------------------------------------
-- Seed: role <-> permission mappings (idempotent via PK)
-- ---------------------------------------------------------------------------
-- root: all permissions
INSERT INTO "app"."sys_role_permissions" ("role_id", "permission_id")
SELECT r."id", p."id"
FROM "app"."sys_roles" r CROSS JOIN "app"."sys_permissions" p
WHERE r."code" = 'root' AND p."type" = 'action'
ON CONFLICT DO NOTHING;

-- tenant_owner: all tenant-scoped actions
INSERT INTO "app"."sys_role_permissions" ("role_id", "permission_id")
SELECT r."id", p."id"
FROM "app"."sys_roles" r CROSS JOIN "app"."sys_permissions" p
WHERE r."code" = 'tenant_owner' AND p."type" = 'action'
ON CONFLICT DO NOTHING;

-- tenant_admin: everything except tenant.delete
INSERT INTO "app"."sys_role_permissions" ("role_id", "permission_id")
SELECT r."id", p."id"
FROM "app"."sys_roles" r CROSS JOIN "app"."sys_permissions" p
WHERE r."code" = 'tenant_admin'
  AND p."type" = 'action'
  AND p."code" <> 'tenant.delete'
ON CONFLICT DO NOTHING;

-- developer: read/create/update keys + read everything else needed to use them
INSERT INTO "app"."sys_role_permissions" ("role_id", "permission_id")
SELECT r."id", p."id"
FROM "app"."sys_roles" r CROSS JOIN "app"."sys_permissions" p
WHERE r."code" = 'developer'
  AND p."code" IN (
    'key.read', 'key.create', 'key.update', 'key.delete',
    'model.read', 'wallet.read', 'task.read', 'task.cancel',
    'tenant.read'
  )
ON CONFLICT DO NOTHING;

-- billing: wallet + read
INSERT INTO "app"."sys_role_permissions" ("role_id", "permission_id")
SELECT r."id", p."id"
FROM "app"."sys_roles" r CROSS JOIN "app"."sys_permissions" p
WHERE r."code" = 'billing'
  AND p."code" IN ('wallet.read', 'wallet.recharge', 'wallet.tx.read', 'tenant.read', 'user.read')
ON CONFLICT DO NOTHING;

-- viewer: read-only across the board
INSERT INTO "app"."sys_role_permissions" ("role_id", "permission_id")
SELECT r."id", p."id"
FROM "app"."sys_roles" r CROSS JOIN "app"."sys_permissions" p
WHERE r."code" = 'viewer'
  AND p."code" LIKE '%.read'
ON CONFLICT DO NOTHING;
