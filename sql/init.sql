CREATE TABLE market_data (
    Time TIMESTAMP WITH TIME ZONE NOT NULL,  -- Временная метка с часовым поясом
    Open NUMERIC(16, 8) NOT NULL,           -- Цена открытия
    High NUMERIC(16, 8) NOT NULL,           -- Максимальная цена
    Low NUMERIC(16, 8) NOT NULL,            -- Минимальная цена
    Close NUMERIC(16, 8) NOT NULL,          -- Цена закрытия
    Volume NUMERIC(24, 12) NOT NULL,        -- Объем
    Symbol TEXT NOT NULL,                   -- Символ (например, BTCUSDT)
    Source TEXT NOT NULL,                   -- Источник данных (например, BINANCE)
    PRIMARY KEY (Time, Symbol, Source)      -- Уникальность записей по времени и символу
);