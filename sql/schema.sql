-- PostgreSQL 스키마 for Medication Agent
-- 테이블 생성

-- 제품 정보 테이블
CREATE TABLE IF NOT EXISTS products (
    item_seq VARCHAR(20) PRIMARY KEY,
    entp_name VARCHAR(200),
    item_name VARCHAR(500),
    item_image TEXT,
    bizrno VARCHAR(20),
    open_de DATE,
    update_de DATE,
    is_otc BOOLEAN DEFAULT false,
    raw_json JSONB
);

-- 제품 섹션 테이블 (효능, 용법, 주의사항 등)
CREATE TABLE IF NOT EXISTS product_sections (
    id SERIAL PRIMARY KEY,
    item_seq VARCHAR(20) REFERENCES products(item_seq) ON DELETE CASCADE,
    section VARCHAR(50) NOT NULL, -- efficacy, dosage, warnings, precautions, interactions, side_effects, storage
    part_idx INTEGER NOT NULL, -- 텍스트 분할 인덱스
    text TEXT NOT NULL,
    UNIQUE(item_seq, section, part_idx)
);

-- 제품 별칭 테이블 (타이레놀, 게보린 등)
CREATE TABLE IF NOT EXISTS product_aliases (
    alias VARCHAR(100),
    item_seq VARCHAR(20) REFERENCES products(item_seq) ON DELETE CASCADE,
    PRIMARY KEY (alias, item_seq)
);

-- 제품 성분 테이블
CREATE TABLE IF NOT EXISTS product_ingredients (
    item_seq VARCHAR(20) REFERENCES products(item_seq) ON DELETE CASCADE,
    ingredient VARCHAR(200),
    PRIMARY KEY (item_seq, ingredient)
);

-- 인덱스 생성
CREATE INDEX IF NOT EXISTS idx_product_sections_item_seq ON product_sections(item_seq);
CREATE INDEX IF NOT EXISTS idx_product_sections_section ON product_sections(section);
CREATE INDEX IF NOT EXISTS idx_product_aliases_alias ON product_aliases(alias);
CREATE INDEX IF NOT EXISTS idx_product_ingredients_ingredient ON product_ingredients(ingredient);

-- JSONB 인덱스
CREATE INDEX IF NOT EXISTS idx_products_raw_json ON products USING GIN (raw_json); 