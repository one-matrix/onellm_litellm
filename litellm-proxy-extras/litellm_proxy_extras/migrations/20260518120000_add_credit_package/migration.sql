-- Add credit_package table in the `app` schema. Editable by admins; the
-- /credit/packages endpoint reads from here for the recharge page.
-- Idempotent so it can re-run safely.

CREATE SCHEMA IF NOT EXISTS "app";

-- CreateTable
-- Note: credits_amount/price_cny/bonus_credits are stored as DOUBLE PRECISION
-- to match the Prisma Float type. prisma-client-py does not generate stable
-- Decimal bindings without the experimental flag, and the rest of the credit
-- subsystem (credit_wallet.gift_balance etc.) already uses double precision.
CREATE TABLE IF NOT EXISTS "app"."credit_package" (
    "id"             UUID NOT NULL DEFAULT gen_random_uuid(),
    "name"           VARCHAR(100) NOT NULL,
    "credits_amount" DOUBLE PRECISION NOT NULL,
    "price_cny"      DOUBLE PRECISION NOT NULL,
    "bonus_credits"  DOUBLE PRECISION DEFAULT 0,
    "badge_text"     VARCHAR(50),
    "is_active"      BOOLEAN NOT NULL DEFAULT TRUE,
    "sort_order"     INTEGER DEFAULT 0,
    "created_at"     TIMESTAMPTZ(6) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at"     TIMESTAMPTZ(6) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "credit_package_pkey" PRIMARY KEY ("id")
);

CREATE INDEX IF NOT EXISTS "credit_package_is_active_idx"
    ON "app"."credit_package"("is_active");

CREATE INDEX IF NOT EXISTS "credit_package_sort_order_idx"
    ON "app"."credit_package"("sort_order");
