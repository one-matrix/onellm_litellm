-- Consolidate tenant-scoped roles.
--
--   tenant_owner + tenant_admin → tenant_admin (full tenant rights, gains
--                                                tenant.delete from owner)
--   developer                   → user         (rename only; permissions
--                                                identical)
--
-- Why: lighter role surface — operators consistently asked "what's the
-- difference between owner and admin?" and the answer was always "delete
-- tenant", which is rare enough not to warrant a separate role. `developer`
-- is renamed to the more natural `user` (the tenant-scoped role; distinct
-- from sys_users.role_global which uses the same string but a different
-- table — no FK between them).
--
-- Idempotent: each step is INSERT … ON CONFLICT DO NOTHING / UPDATE WHERE /
-- DELETE WHERE, safe to re-run on a partially-applied DB.

-- 1. Grant tenant_admin the full tenant-scope permission set (it was
--    previously missing tenant.delete — the only thing that separated it
--    from tenant_owner).
INSERT INTO "app"."sys_role_permissions" ("role_id", "permission_id")
SELECT r."id", p."id"
FROM "app"."sys_roles" r CROSS JOIN "app"."sys_permissions" p
WHERE r."code" = 'tenant_admin' AND p."type" = 'action'
ON CONFLICT DO NOTHING;

-- 2. Re-point every (user, tenant_owner, tenant) membership onto
--    tenant_admin. ON CONFLICT swallows the case where the user already
--    held both roles in the same tenant.
INSERT INTO "app"."sys_user_roles" ("user_id", "role_id", "tenant_id", "created_at")
SELECT ur."user_id", new_role."id", ur."tenant_id", ur."created_at"
FROM "app"."sys_user_roles" ur
JOIN "app"."sys_roles" old_role
  ON ur."role_id" = old_role."id" AND old_role."code" = 'tenant_owner'
JOIN "app"."sys_roles" new_role
  ON new_role."code" = 'tenant_admin'
ON CONFLICT DO NOTHING;

-- 3. Drop the now-unused tenant_owner role. Cascades remove the original
--    sys_user_roles rows (step 2 has already cloned them) and the
--    role_permissions rows.
DELETE FROM "app"."sys_roles" WHERE "code" = 'tenant_owner';

-- 4. Rename developer → user. Affects sys_roles.code only; sys_user_roles
--    rows keep their role_id (a UUID) so member assignments survive
--    unchanged. The check constraint on sys_users.role_global is unrelated
--    (different table, different field).
UPDATE "app"."sys_roles"
SET "code" = 'user',
    "name" = 'User',
    "normalized_name" = 'USER',
    "description" = 'Create and use API keys; read models/wallet/tasks'
WHERE "code" = 'developer';
