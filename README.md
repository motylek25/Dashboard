# Дашборд по SampleSuperstore (Streamlit)

Интерактивный дашборд на Streamlit + Plotly. Работает с разными вариантами CSV (автодетект кодировки/разделителя, гибкие имена колонок).
Если нет колонки `Order Date`, дашборд работает без временной аналитики.

## Возможности
- Фильтры: Region / Segment / Category (+ даты, если есть `Order Date`).
- KPI: Sales, Profit, Items, Avg Discount, Profit margin.
- Графики:
  - Динамика по месяцам (Sales/Profit, опционально маржа).
  - Продажи по категориям и топ‑подкатегории.
  - Топ по прибыли (Product Name → fallback на Sub‑Category → Category).
  - Топ штатов по продажам.
  - Scatter: Discount vs Profit.
- Экспорт отфильтрованных данных в CSV.

## Запуск
Положите `SampleSuperstore.csv` рядом с `app.py` или загрузите через загрузчик в сайдбаре.

Дашборд доступен на http://localhost:8501/

