-- Delta Airlines ML Platform — PostgreSQL Init
-- Schema: predictions, model_registry, monitoring

CREATE SCHEMA IF NOT EXISTS delta_ml;

-- ─── Table prédictions ────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS delta_ml.predictions (
    id                    SERIAL PRIMARY KEY,
    created_at            TIMESTAMP DEFAULT NOW(),
    carrier               VARCHAR(5)  DEFAULT 'DL',
    origin                VARCHAR(5)  NOT NULL,
    dest                  VARCHAR(5)  NOT NULL,
    route                 VARCHAR(12) NOT NULL,
    year                  INTEGER     NOT NULL,
    month                 INTEGER     NOT NULL,
    day_of_week           INTEGER,
    seats                 INTEGER,
    avg_ticket_price      FLOAT,
    weather_condition     VARCHAR(10),
    is_holiday_period     BOOLEAN,
    predicted_load_factor FLOAT       NOT NULL,
    predicted_passengers  INTEGER,
    estimated_revenue     FLOAT,
    confidence_low        FLOAT,
    confidence_high       FLOAT,
    performance_rating    VARCHAR(15),
    model_used            VARCHAR(50),
    mae_model             FLOAT,
    request_source        VARCHAR(50) DEFAULT 'api'
);

-- ─── Table model registry ─────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS delta_ml.model_registry (
    id             SERIAL PRIMARY KEY,
    registered_at  TIMESTAMP DEFAULT NOW(),
    model_name     VARCHAR(50)  NOT NULL,
    version        VARCHAR(20)  NOT NULL,
    mae            FLOAT,
    rmse           FLOAT,
    r2             FLOAT,
    mape           FLOAT,
    n_features     INTEGER,
    train_samples  INTEGER,
    mlflow_run_id  VARCHAR(100),
    is_production  BOOLEAN DEFAULT FALSE,
    carrier        VARCHAR(5) DEFAULT 'DL'
);

-- ─── Table monitoring ─────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS delta_ml.monitoring_logs (
    id              SERIAL PRIMARY KEY,
    logged_at       TIMESTAMP DEFAULT NOW(),
    carrier         VARCHAR(5) DEFAULT 'DL',
    metric_name     VARCHAR(50),
    metric_value    FLOAT,
    route           VARCHAR(12),
    month           INTEGER,
    alert_triggered BOOLEAN DEFAULT FALSE,
    alert_message   VARCHAR(200)
);

-- ─── Indexes ──────────────────────────────────────────────────────────────────
CREATE INDEX IF NOT EXISTS idx_pred_route
    ON delta_ml.predictions(route);
CREATE INDEX IF NOT EXISTS idx_pred_created
    ON delta_ml.predictions(created_at);
CREATE INDEX IF NOT EXISTS idx_pred_carrier
    ON delta_ml.predictions(carrier);

-- ─── Seed model registry ──────────────────────────────────────────────────────
INSERT INTO delta_ml.model_registry
    (model_name, version, mae, r2, n_features, is_production, carrier)
VALUES
    ('LightGBM',        'v1.0', 0.363, 0.9991, 63, TRUE,  'DL'),
    ('XGBoost',         'v1.0', 0.414, 0.9989, 63, FALSE, 'DL'),
    ('GradientBoosting','v1.0', 0.504, 0.9983, 63, FALSE, 'DL'),
    ('RandomForest',    'v1.0', 2.774, 0.9561, 63, FALSE, 'DL');

SELECT 'Delta Airlines ML Database initialized successfully' AS status;