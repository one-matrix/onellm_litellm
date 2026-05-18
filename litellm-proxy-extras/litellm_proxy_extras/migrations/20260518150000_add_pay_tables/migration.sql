-- Generic payment platform — 3 tables in the `app` schema. Channel-agnostic
-- (alipay/wechat/stripe/...) and product-agnostic (credits/subscription/...).
-- Mirrors docs/sql/pay.sql with two adaptations for the LiteLLM stack:
--   * tenant_id/user_id stored as TEXT (matches LiteLLM_TeamTable.team_id style)
--   * amount columns are DOUBLE PRECISION (prisma-client-py does not support
--     NUMERIC/Decimal without the experimental flag; consistent with the rest
--     of the credit subsystem)
-- Idempotent so it can re-run safely.

CREATE SCHEMA IF NOT EXISTS "app";

-- 1. pay_order ---------------------------------------------------------------
CREATE TABLE IF NOT EXISTS "app"."pay_order" (
    "id"                 UUID NOT NULL DEFAULT gen_random_uuid(),
    "out_trade_no"       VARCHAR(64) NOT NULL,
    "channel"            VARCHAR(20) NOT NULL,
    "channel_trade_no"   VARCHAR(128),

    "tenant_id"          TEXT,
    "user_id"            TEXT NOT NULL,

    "product_type"       VARCHAR(40) NOT NULL,
    "product_id"         VARCHAR(64),
    "product_name"       VARCHAR(256) NOT NULL,
    "quantity"           INTEGER NOT NULL DEFAULT 1,
    "unit"               VARCHAR(20),
    "billing_cycle"      VARCHAR(40),
    "product_meta"       JSONB NOT NULL DEFAULT '{}',

    "amount_cny"         DOUBLE PRECISION NOT NULL,
    "paid_amount_cny"    DOUBLE PRECISION,
    "currency"           VARCHAR(3) NOT NULL DEFAULT 'CNY',

    "status"             VARCHAR(20) NOT NULL DEFAULT 'pending',
    "paid_at"            TIMESTAMPTZ(6),
    "expire_at"          TIMESTAMPTZ(6) NOT NULL,

    "pay_account"        VARCHAR(128),
    "financial_category" VARCHAR(40),

    "remark"             VARCHAR(500),
    "client_ip"          VARCHAR(45),
    "metadata"           JSONB NOT NULL DEFAULT '{}',

    "created_at"         TIMESTAMPTZ(6) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "created_by"         TEXT,
    "updated_at"         TIMESTAMPTZ(6) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_by"         TEXT,
    "is_deleted"         BOOLEAN NOT NULL DEFAULT FALSE,
    "deleted_at"         TIMESTAMPTZ(6),
    "deleted_by"         TEXT,

    CONSTRAINT "pay_order_pkey" PRIMARY KEY ("id"),
    CONSTRAINT "chk_pay_order_amount_positive" CHECK ("amount_cny" > 0),
    CONSTRAINT "chk_pay_order_qty_positive"    CHECK ("quantity" > 0)
);

CREATE UNIQUE INDEX IF NOT EXISTS "pay_order_out_trade_no_key" ON "app"."pay_order"("out_trade_no");
CREATE INDEX IF NOT EXISTS "pay_order_tenant_id_idx"           ON "app"."pay_order"("tenant_id");
CREATE INDEX IF NOT EXISTS "pay_order_user_id_idx"             ON "app"."pay_order"("user_id");
CREATE INDEX IF NOT EXISTS "pay_order_status_idx"              ON "app"."pay_order"("status");
CREATE INDEX IF NOT EXISTS "pay_order_created_at_idx"          ON "app"."pay_order"("created_at" DESC);
CREATE INDEX IF NOT EXISTS "pay_order_channel_trade_no_idx"    ON "app"."pay_order"("channel_trade_no");
CREATE INDEX IF NOT EXISTS "pay_order_product_type_id_idx"     ON "app"."pay_order"("product_type", "product_id");

-- 2. pay_notify_log ----------------------------------------------------------
CREATE TABLE IF NOT EXISTS "app"."pay_notify_log" (
    "id"               UUID NOT NULL DEFAULT gen_random_uuid(),
    "channel"          VARCHAR(20) NOT NULL,
    "order_id"         UUID,
    "out_trade_no"     VARCHAR(64),
    "channel_trade_no" VARCHAR(128),
    "trade_status"     VARCHAR(40),
    "raw_body"         JSONB NOT NULL,
    "signature_valid"  BOOLEAN NOT NULL,
    "processed"        BOOLEAN NOT NULL DEFAULT FALSE,
    "error_message"    TEXT,
    "client_ip"        VARCHAR(45),
    "received_at"      TIMESTAMPTZ(6) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "pay_notify_log_pkey" PRIMARY KEY ("id")
);

CREATE INDEX IF NOT EXISTS "pay_notify_log_out_trade_no_idx" ON "app"."pay_notify_log"("out_trade_no");
CREATE INDEX IF NOT EXISTS "pay_notify_log_order_id_idx"     ON "app"."pay_notify_log"("order_id");
CREATE INDEX IF NOT EXISTS "pay_notify_log_received_at_idx"  ON "app"."pay_notify_log"("received_at" DESC);

-- 3. pay_refund --------------------------------------------------------------
CREATE TABLE IF NOT EXISTS "app"."pay_refund" (
    "id"                 UUID NOT NULL DEFAULT gen_random_uuid(),

    "order_id"           UUID NOT NULL,
    "out_trade_no"       VARCHAR(64) NOT NULL,
    "out_refund_no"      VARCHAR(64) NOT NULL,
    "channel"            VARCHAR(20) NOT NULL,
    "channel_refund_no"  VARCHAR(128),

    "refund_amount_cny"  DOUBLE PRECISION NOT NULL,
    "currency"           VARCHAR(3) NOT NULL DEFAULT 'CNY',

    "reason"             VARCHAR(500) NOT NULL DEFAULT '',
    "status"             VARCHAR(20) NOT NULL DEFAULT 'pending',
    "error_message"      TEXT,

    "business_settled"   BOOLEAN NOT NULL DEFAULT FALSE,
    "settlement_meta"    JSONB NOT NULL DEFAULT '{}',

    "operator_id"        TEXT NOT NULL,
    "refunded_at"        TIMESTAMPTZ(6),
    "created_at"         TIMESTAMPTZ(6) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "created_by"         TEXT,
    "updated_at"         TIMESTAMPTZ(6) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_by"         TEXT,
    "is_deleted"         BOOLEAN NOT NULL DEFAULT FALSE,
    "deleted_at"         TIMESTAMPTZ(6),
    "deleted_by"         TEXT,

    CONSTRAINT "pay_refund_pkey" PRIMARY KEY ("id"),
    CONSTRAINT "chk_pay_refund_amount_positive" CHECK ("refund_amount_cny" > 0),
    CONSTRAINT "pay_refund_order_fkey" FOREIGN KEY ("order_id") REFERENCES "app"."pay_order"("id") ON DELETE RESTRICT ON UPDATE CASCADE
);

CREATE UNIQUE INDEX IF NOT EXISTS "pay_refund_out_refund_no_key" ON "app"."pay_refund"("out_refund_no");
CREATE INDEX IF NOT EXISTS "pay_refund_order_id_idx"             ON "app"."pay_refund"("order_id");
CREATE INDEX IF NOT EXISTS "pay_refund_out_trade_no_idx"         ON "app"."pay_refund"("out_trade_no");
CREATE INDEX IF NOT EXISTS "pay_refund_status_idx"               ON "app"."pay_refund"("status");
CREATE INDEX IF NOT EXISTS "pay_refund_created_at_idx"           ON "app"."pay_refund"("created_at" DESC);
