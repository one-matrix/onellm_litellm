-- Credit wallet tables live in the `app` schema, separate from the litellm
-- core `public` schema, so upstream migrations never collide with onellm
-- extensions. Both the schema and the enums are created idempotently so
-- this migration can re-run safely (e.g. during a redeploy on a DB where a
-- previous attempt partially landed).

-- CreateSchema
CREATE SCHEMA IF NOT EXISTS "app";

-- CreateEnum
DO $$ BEGIN
  CREATE TYPE "app"."CreditTxType" AS ENUM (
    'purchase',
    'gift',
    'consumption',
    'pre_deduct',
    'settlement',
    'refund',
    'expiration',
    'admin_adjust'
  );
EXCEPTION WHEN duplicate_object THEN null; END $$;

-- CreateEnum
DO $$ BEGIN
  CREATE TYPE "app"."CreditWalletType" AS ENUM ('gift', 'paid');
EXCEPTION WHEN duplicate_object THEN null; END $$;

-- CreateTable: per-tenant credit wallet (frozen / gift / paid balances + lifetime stats)
CREATE TABLE IF NOT EXISTS "app"."credit_wallet" (
    "id"                    TEXT NOT NULL,
    "tenant_id"             TEXT NOT NULL,
    "gift_balance"          DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    "paid_balance"          DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    "frozen_amount"         DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    "total_consumed"        DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    "total_recharged"       DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    "total_gifted"          DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    "low_balance_threshold" DOUBLE PRECISION,
    "created_at"            TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at"            TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "credit_wallet_pkey" PRIMARY KEY ("id")
);

CREATE UNIQUE INDEX IF NOT EXISTS "credit_wallet_tenant_id_key"
    ON "app"."credit_wallet"("tenant_id");

CREATE INDEX IF NOT EXISTS "credit_wallet_tenant_id_idx"
    ON "app"."credit_wallet"("tenant_id");

-- CreateTable: append-only credit transaction log
CREATE TABLE IF NOT EXISTS "app"."credit_transaction" (
    "id"              TEXT NOT NULL,
    "tenant_id"       TEXT NOT NULL,
    "tx_type"         "app"."CreditTxType" NOT NULL,
    "wallet_type"     "app"."CreditWalletType" NOT NULL,
    "amount"          DOUBLE PRECISION NOT NULL,
    "balance_after"   DOUBLE PRECISION NOT NULL,
    "agent_record_id" TEXT,
    "model_name"      TEXT,
    "description"     TEXT,
    "metadata"        JSONB NOT NULL DEFAULT '{}',
    "created_at"      TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "credit_transaction_pkey" PRIMARY KEY ("id")
);

CREATE INDEX IF NOT EXISTS "credit_transaction_tenant_id_idx"
    ON "app"."credit_transaction"("tenant_id");

CREATE INDEX IF NOT EXISTS "credit_transaction_agent_record_id_idx"
    ON "app"."credit_transaction"("agent_record_id");

CREATE INDEX IF NOT EXISTS "credit_transaction_created_at_idx"
    ON "app"."credit_transaction"("created_at");
