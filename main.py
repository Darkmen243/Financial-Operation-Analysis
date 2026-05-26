import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression

# 1. ЗАГРУЗКА И ОЧИСТКА ДАННЫХ
transactions = pd.read_excel('transactions_data.xlsx')
with open('clients_data.json', 'r') as f:
    clients = pd.DataFrame(json.load(f))
if 'id' in clients.columns:
    clients.rename(columns={'id': 'client_id'}, inplace=True)

# Очистка транзакций
transactions.dropna(subset=['transaction_id', 'amount', 'transaction_date'], inplace=True)
transactions['client_id'] = transactions['client_id'].fillna('Unknown')
transactions['payment_method'] = transactions['payment_method'].fillna('Unknown')
transactions['city'] = transactions['city'].fillna('Unknown')
transactions['service'] = transactions['service'].fillna('other')
transactions = transactions[transactions['amount'] >= 0]
transactions['transaction_date'] = pd.to_datetime(transactions['transaction_date'], errors='coerce')
transactions.dropna(subset=['transaction_date'], inplace=True)

# Очистка клиентов
clients.dropna(subset=['client_id', 'net_worth'], inplace=True)
if 'age' in clients.columns:
    clients['age'] = pd.to_numeric(clients['age'], errors='coerce')
    clients = clients[clients['age'].notna()]
    clients = clients[(clients['age'] >= 0) & (clients['age'] <= 120)]
clients = clients[clients['net_worth'] >= 0]
clients = clients.drop_duplicates(subset=['client_id'], keep='first')

# 2. АНАЛИЗ ДАННЫХ
print("Топ-5 услуг по количеству заказов:")
print(transactions['service'].value_counts().head(5))

print("\nСредняя сумма транзакций по городам:")
print(transactions.groupby('city')['amount'].mean().sort_values(ascending=False))

print(f"\nУслуга с наибольшей выручкой: {transactions.groupby('service')['amount'].sum().idxmax()}")

print("\nПроцент транзакций по способам оплаты:")
print(transactions['payment_method'].value_counts(normalize=True) * 100)

max_date = transactions['transaction_date'].max()
last_month_revenue = transactions[transactions['transaction_date'] >= (max_date - pd.DateOffset(months=1))]['amount'].sum()
print(f"\nВыручка за последний месяц: {last_month_revenue:,.2f}")

# 3. ОБЪЕДИНЕНИЕ И УРОВНИ АКТИВОВ
merged = transactions.merge(clients, on='client_id', how='left')

def wealth_level(nw):
    if pd.isna(nw): return 'Нет данных'
    elif nw < 100000: return 'Низкий капитал'
    elif nw <= 1000000: return 'Средний капитал'
    else: return 'Высокий капитал'

merged['wealth_level'] = merged['net_worth'].apply(wealth_level)
print(f"\nНаибольшую выручку приносят: {merged.groupby('wealth_level')['amount'].sum().idxmax()}")

# 4. ВИЗУАЛИЗАЦИЯ
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Распределение сумм
transactions[transactions['amount'] <= transactions['amount'].quantile(0.95)]['amount'].hist(ax=axes[0,0], bins=50)
axes[0,0].set_title('Распределение сумм транзакций')

# Выручка по услугам
transactions.groupby('service')['amount'].sum().head(8).plot(kind='barh', ax=axes[0,1])
axes[0,1].set_title('Выручка по услугам')

# Зависимость от возраста
merged[merged['age'].notna()].groupby('age')['amount'].mean().plot(ax=axes[1,0])
axes[1,0].set_title('Средний чек по возрастам')

# Способы оплаты
transactions['payment_method'].value_counts().plot(kind='pie', ax=axes[1,1], autopct='%1.1f%%')
axes[1,1].set_title('Способы оплаты')

plt.tight_layout()
plt.show()

# 5. УЛУЧШЕННОЕ ПРОГНОЗИРОВАНИЕ
print("\n5. ПРОГНОЗИРОВАНИЕ СПРОСА НА СЛЕДУЮЩИЙ МЕСЯЦ")
print("-"*40)

# Агрегация по дням
daily = transactions.groupby(transactions['transaction_date'].dt.date)['amount'].sum().reset_index()
daily.columns = ['date', 'revenue']
daily['date'] = pd.to_datetime(daily['date'])
daily = daily.sort_values('date')

if len(daily) >= 30:
    # СОЗДАНИЕ РАСШИРЕННЫХ ПРИЗНАКОВ
    daily['day_of_week'] = daily['date'].dt.dayofweek  # 0=пн, 6=вс
    daily['month'] = daily['date'].dt.month
    daily['weekend'] = (daily['day_of_week'] >= 5).astype(int)  # выходной?
    
    # ЛАГОВЫЕ ПРИЗНАКИ (значения за предыдущие дни)
    daily['revenue_lag1'] = daily['revenue'].shift(1)  # вчера
    daily['revenue_lag7'] = daily['revenue'].shift(7)  # неделю назад
    
    # СКОЛЬЗЯЩИЕ СРЕДНИЕ
    daily['revenue_ma7'] = daily['revenue'].rolling(window=7, min_periods=1).mean()
    
    # Удаляем строки с NaN (из-за лагов)
    daily_clean = daily.dropna()
     
    # Полный набор признаков
    features_full = ['day_of_week', 'month', 'weekend', 'revenue_lag1', 'revenue_lag7', 'revenue_ma7']
    selected_features = features_full
    
    X = daily_clean[selected_features].values
    y = daily_clean['revenue'].values
    
    # РАЗДЕЛЕНИЕ НА ТРЕНИРОВКУ И ТЕСТ
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # ОБУЧЕНИЕ МОДЕЛИ
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_absolute_error, r2_score
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # ОЦЕНКА КАЧЕСТВА
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Качество модели:")
    print(f"  - MAE: {mae:,.2f} (средняя ошибка в день)")
    print(f"  - R²: {r2:.3f} (чем ближе к 1, тем лучше)")
    
    # ПРОГНОЗ НА СЛЕДУЮЩИЙ МЕСЯЦ
    last_date = daily['date'].max()
    future_dates = [last_date + timedelta(days=i) for i in range(1, 31)]
    
    # Создаем признаки для будущих дней
    future_data = []
    last_revenues = daily['revenue'].tail(30).values  # последние 30 значений
    
    for i, d in enumerate(future_dates):
        # Базовые признаки
        day_of_week = d.dayofweek
        month = d.month
        weekend = 1 if day_of_week >= 5 else 0
        
        # Лаговые признаки (на основе последних известных значений)
        revenue_lag1 = last_revenues[-1] if len(last_revenues) > 0 else 0
        revenue_lag7 = last_revenues[-7] if len(last_revenues) >= 7 else revenue_lag1
        revenue_ma7 = np.mean(last_revenues[-7:]) if len(last_revenues) >= 7 else revenue_lag1
        
        future_data.append([day_of_week, month, weekend, revenue_lag1, revenue_lag7, revenue_ma7])
        
        # Обновляем для следующего дня (прогнозируем рекурсивно)
        simple_forecast = model.predict([future_data[-1]])[0]
        last_revenues = np.append(last_revenues, simple_forecast)
    
    forecast = model.predict(future_data)
    forecast = np.maximum(forecast, 0)  # неотрицательные значения
    
    print(f"\nПрогноз на следующий месяц ({future_dates[0].strftime('%Y-%m-%d')} - {future_dates[-1].strftime('%Y-%m-%d')}):")
    print(f"  - Суммарная выручка: {forecast.sum():,.2f}")
    print(f"  - Средний дневной доход: {forecast.mean():,.2f}")
    print(f"  - Минимальный прогноз: {forecast.min():,.2f}")
    print(f"  - Максимальный прогноз: {forecast.max():,.2f}")
    
    # Сравнение с прошлым месяцем
    prev_month_revenue = daily[daily['date'] >= (last_date - pd.DateOffset(months=1))]['revenue'].sum()
    change_pct = ((forecast.sum() - prev_month_revenue) / prev_month_revenue) * 100
    print(f"\nСравнение с прошлым месяцем:")
    print(f"  - Было: {prev_month_revenue:,.2f}")
    print(f"  - Будет: {forecast.sum():,.2f}")
    print(f"  - Изменение: {change_pct:+.1f}%")
    
    plt.figure(figsize=(12, 5))
    plt.plot(daily['date'], daily['revenue'], label='Факт', linewidth=2)
    plt.plot(future_dates, forecast, 'r--', label='Прогноз', linewidth=2, marker='o', markersize=3)
    plt.axvline(x=last_date, color='gray', linestyle='--', alpha=0.5)
    plt.xlabel('Дата')
    plt.ylabel('Выручка')
    plt.title('Прогноз выручки на следующий месяц')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()  
else:
    print("  Недостаточно данных для прогнозирования (нужно минимум 30 дней)")