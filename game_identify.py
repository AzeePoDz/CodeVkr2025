import numpy as np
import matplotlib.pyplot as plt
import random
import time
import pandas as pd
import os
from typing import List, Tuple, Dict, Any
from dataclasses import dataclass


def load_attack_data_from_dataset(dataset_path):
    """
    Загрузка данных об атаках из набора данных

    Args:
        dataset_path: Путь к файлу с данными об атаках

    Returns:
        dict: Словарь с информацией о векторах атак
    """
    df = pd.read_csv(dataset_path)

    attack_data = {
        'names': [],
        'damage_potential': [],
        'cost': [],
        'max_investment': []
    }

    # Извлечение релевантной информации
    for _, row in df.iterrows():
        attack_data['names'].append(row['attack_name'])
        attack_data['damage_potential'].append(row['estimated_damage'])
        attack_data['cost'].append(row['execution_cost'])
        attack_data['max_investment'].append(row['max_investment'])

    return attack_data


def load_defense_data_from_dataset(dataset_path):
    """
    Загрузка данных о механизмах защиты из набора данных

    Args:
        dataset_path: Путь к файлу с данными о защите

    Returns:
        dict: Словарь с информацией о механизмах защиты
    """
    df = pd.read_csv(dataset_path)

    defense_data = {
        'names': [],
        'cost': [],
        'max_investment': []
    }

    # Извлечение релевантной информации
    for _, row in df.iterrows():
        defense_data['names'].append(row['defense_name'])
        defense_data['cost'].append(row['implementation_cost'])
        defense_data['max_investment'].append(row['max_investment'])

    return defense_data


def load_effectiveness_matrix(dataset_path, attack_names, defense_names):
    """
    Загрузка матрицы эффективности из набора данных

    Args:
        dataset_path: Путь к файлу с данными об эффективности
        attack_names: Список имен атак
        defense_names: Список имен механизмов защиты

    Returns:
        numpy.ndarray: Матрица вероятностей эффективности
    """
    df = pd.read_csv(dataset_path)

    # Создание отображения имен на индексы
    attack_indices = {name: i for i, name in enumerate(attack_names)}
    defense_indices = {name: j for j, name in enumerate(defense_names)}

    # Инициализация матрицы нулями
    n = len(attack_names)
    m = len(defense_names)
    P = np.zeros((n, m))

    # Заполнение матрицы на основе набора данных
    for _, row in df.iterrows():
        attack_name = row['attack_name']
        defense_name = row['defense_name']
        effectiveness = row['effectiveness']

        if attack_name in attack_indices and defense_name in defense_indices:
            i = attack_indices[attack_name]
            j = defense_indices[defense_name]
            P[i, j] = effectiveness

    return P


def process_cicddos2019_dataset(dataset_dir, output_path):
    """
    Обработка набора данных CIC-DDoS2019 из нескольких файлов для создания профилей атак

    Args:
        dataset_dir: Путь к директории с файлами CIC-DDoS2019
        output_path: Путь для сохранения обработанных данных
    """
    import glob

    attack_data = {
        'attack_name': [],
        'estimated_damage': [],
        'execution_cost': [],
        'max_investment': []
    }

    # Получаем список всех CSV-файлов в директории
    csv_files = glob.glob(os.path.join(dataset_dir, "*.csv"))

    if not csv_files:
        print(f"Ошибка: CSV-файлы не найдены в {dataset_dir}")
        return

    print(f"Найдено {len(csv_files)} CSV-файлов для обработки.")

    for csv_file in csv_files:
        # Извлекаем имя атаки из имени файла
        attack_type = os.path.basename(csv_file).split('.')[0]
        print(f"Обработка файла {attack_type}...")

        try:
            # Читаем CSV-файл
            df = pd.read_csv(csv_file, low_memory=False)

            # Обработка данных из файла
            # Расчет потенциального ущерба на основе статистики потоков
            if 'Total Fwd Packets' in df.columns:
                total_packets = df['Total Fwd Packets'].sum()
            elif 'Total Packets' in df.columns:
                total_packets = df['Total Packets'].sum()
            else:
                # Альтернативные метрики, если нужных колонок нет
                total_packets = len(df) * 100

            # Нормализация к денежному значению (произвольное масштабирование)
            damage_potential = total_packets / 1e6 * 1000  # $1000 на миллион пакетов

            # Оценка стоимости выполнения атаки на основе сложности
            # Например, используем количество уникальных IP-адресов источника
            if 'Source IP' in df.columns:
                unique_sources = df['Source IP'].nunique()
            elif ' Source IP' in df.columns:  # некоторые файлы могут иметь пробел перед именем столбца
                unique_sources = df[' Source IP'].nunique()
            else:
                # Если нет нужной колонки, используем приближение
                unique_sources = int(len(df) / 100) + 1

            execution_cost = unique_sources * 100  # $100 за IP-адрес источника

            # Установка максимальных инвестиций как функции от стоимости выполнения
            max_investment = execution_cost * 2

            # Добавление в данные
            attack_data['attack_name'].append(attack_type)
            attack_data['estimated_damage'].append(damage_potential)
            attack_data['execution_cost'].append(execution_cost)
            attack_data['max_investment'].append(max_investment)

            print(f"  - Обработано записей: {len(df)}")
            print(f"  - Потенциальный ущерб: ${damage_potential:.2f}")
            print(f"  - Стоимость выполнения: ${execution_cost:.2f}")

        except Exception as e:
            print(f"Ошибка при обработке файла {csv_file}: {str(e)}")

    # Сохранение обработанных данных
    pd.DataFrame(attack_data).to_csv(output_path, index=False)
    print(f"Обработано {len(attack_data['attack_name'])} типов атак и сохранено в {output_path}")


def create_defense_dataset(output_path):
    """
    Создание синтетического набора данных о механизмах защиты на основе известных
    средств защиты от DDoS-атак

    Args:
        output_path: Путь для сохранения данных о защите

    Returns:
        list: Список имен механизмов защиты
    """
    # Определение распространенных механизмов защиты от DDoS
    defenses = [
        {
            'defense_name': 'Ограничение скорости (Rate Limiting)',
            'implementation_cost': 5000,
            'max_investment': 15000,
            'description': 'Ограничивает скорость входящих запросов'
        },
        {
            'defense_name': 'IP-фильтрация',
            'implementation_cost': 3000,
            'max_investment': 10000,
            'description': 'Блокирует трафик с подозрительных IP-адресов'
        },
        {
            'defense_name': 'Анализ трафика',
            'implementation_cost': 12000,
            'max_investment': 30000,
            'description': 'Использует машинное обучение для выявления вредоносных шаблонов трафика'
        },
        {
            'defense_name': 'CDN-защита',
            'implementation_cost': 8000,
            'max_investment': 25000,
            'description': 'Распределяет трафик по нескольким серверам'
        },
        {
            'defense_name': 'Межсетевой экран уровня приложений',
            'implementation_cost': 15000,
            'max_investment': 40000,
            'description': 'Фильтрует трафик на основе правил уровня приложений'
        },
        {
            'defense_name': 'Anycast-сеть',
            'implementation_cost': 25000,
            'max_investment': 60000,
            'description': 'Распределяет трафик по нескольким глобальным точкам присутствия'
        }
    ]

    # Сохранение данных о защите
    pd.DataFrame(defenses).to_csv(output_path, index=False)
    print(f"Создан набор данных с {len(defenses)} механизмами защиты и сохранен в {output_path}")

    return [d['defense_name'] for d in defenses]


def create_effectiveness_matrix(output_path, attack_names, defense_names):
    """
    Создание синтетического набора данных значений эффективности
    для каждой пары атака-защита

    Args:
        output_path: Путь для сохранения данных об эффективности
        attack_names: Список имен атак
        defense_names: Список имен механизмов защиты
    """
    effectiveness_data = {
        'attack_name': [],
        'defense_name': [],
        'effectiveness': []
    }

       effectiveness_rules = {
        'Ограничение скорости (Rate Limiting)': {
            'UDP Flood': 0.75,
            'TCP SYN Flood': 0.70,
            'HTTP Flood': 0.60,
            'DNS Amplification': 0.40,
            'NTP Amplification': 0.35,
            'LDAP': 0.45,
            'MSSQL': 0.30,
            'NetBIOS': 0.55,
            'SNMP': 0.35,
            'SSDP': 0.40,
            'WebDDoS': 0.65,
            'TFTP': 0.40
        },
        'IP-фильтрация': {
            'UDP Flood': 0.60,
            'TCP SYN Flood': 0.65,
            'HTTP Flood': 0.30,
            'DNS Amplification': 0.80,
            'NTP Amplification': 0.75,
            'LDAP': 0.70,
            'MSSQL': 0.65,
            'NetBIOS': 0.70,
            'SNMP': 0.75,
            'SSDP': 0.75,
            'WebDDoS': 0.40,
            'TFTP': 0.65
        },
        'Анализ трафика': {
            'UDP Flood': 0.85,
            'TCP SYN Flood': 0.80,
            'HTTP Flood': 0.75,
            'DNS Amplification': 0.70,
            'NTP Amplification': 0.70,
            'LDAP': 0.80,
            'MSSQL': 0.75,
            'NetBIOS': 0.75,
            'SNMP': 0.80,
            'SSDP': 0.75,
            'WebDDoS': 0.70,
            'TFTP': 0.75
        },
        'CDN-защита': {
            'UDP Flood': 0.40,
            'TCP SYN Flood': 0.45,
            'HTTP Flood': 0.85,
            'DNS Amplification': 0.30,
            'NTP Amplification': 0.25,
            'LDAP': 0.30,
            'MSSQL': 0.20,
            'NetBIOS': 0.25,
            'SNMP': 0.30,
            'SSDP': 0.25,
            'WebDDoS': 0.85,
            'TFTP': 0.25
        },
        'Межсетевой экран уровня приложений': {
            'UDP Flood': 0.30,
            'TCP SYN Flood': 0.65,
            'HTTP Flood': 0.90,
            'DNS Amplification': 0.20,
            'NTP Amplification': 0.15,
            'LDAP': 0.25,
            'MSSQL': 0.75,
            'NetBIOS': 0.20,
            'SNMP': 0.15,
            'SSDP': 0.20,
            'WebDDoS': 0.85,
            'TFTP': 0.25
        },
        'Anycast-сеть': {
            'UDP Flood': 0.70,
            'TCP SYN Flood': 0.75,
            'HTTP Flood': 0.80,
            'DNS Amplification': 0.85,
            'NTP Amplification': 0.80,
            'LDAP': 0.75,
            'MSSQL': 0.65,
            'NetBIOS': 0.70,
            'SNMP': 0.75,
            'SSDP': 0.80,
            'WebDDoS': 0.85,
            'TFTP': 0.75
        }
    }

    # Генерация значений эффективности для всех пар
    for attack in attack_names:
        for defense in defense_names:
            # Если есть предопределенное правило, используем его
            if defense in effectiveness_rules and attack in effectiveness_rules[defense]:
                effectiveness = effectiveness_rules[defense][attack]
            else:
                # Иначе генерируем случайное значение с разумным распределением
                effectiveness = random.betavariate(2, 2)  # Генерирует значения, центрированные около 0.5

            effectiveness_data['attack_name'].append(attack)
            effectiveness_data['defense_name'].append(defense)
            effectiveness_data['effectiveness'].append(effectiveness)

    # Сохранение данных об эффективности
    pd.DataFrame(effectiveness_data).to_csv(output_path, index=False)
    print(
        f"Создана матрица эффективности с {len(effectiveness_data['attack_name'])} записями и сохранена в {output_path}")


class GameParameters:
    """Класс, содержащий параметры дискретно-непрерывной игры."""

    def __init__(self, n_attacks=None, m_defenses=None, seed=None,
                 attack_dataset=None, defense_dataset=None, effectiveness_dataset=None):
        """
        Инициализация параметров игры либо случайными значениями, либо из наборов данных.

        Args:
            n_attacks: Количество возможных векторов атаки (используется только если не загружается из набора данных)
            m_defenses: Количество возможных механизмов защиты (используется только если не загружается из набора данных)
            seed: Случайное зерно для воспроизводимости
            attack_dataset: Путь к набору данных с информацией об атаках
            defense_dataset: Путь к набору данных с информацией о защите
            effectiveness_dataset: Путь к набору данных с матрицей эффективности
        """
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        if attack_dataset and defense_dataset and effectiveness_dataset:
            self.initialize_from_datasets(attack_dataset, defense_dataset, effectiveness_dataset)
        else:
            self.n = n_attacks
            self.m = m_defenses
            self.N = list(range(self.n))
            self.M = list(range(self.m))
            self.initialize_parameters()

    def initialize_from_datasets(self, attack_dataset, defense_dataset, effectiveness_dataset):
        """Инициализация параметров из наборов данных"""
        # Загрузка данных об атаках
        attack_data = load_attack_data_from_dataset(attack_dataset)
        self.attack_names = attack_data['names']
        self.n = len(self.attack_names)
        self.N = list(range(self.n))

        # Загрузка данных о защите
        defense_data = load_defense_data_from_dataset(defense_dataset)
        self.defense_names = defense_data['names']
        self.m = len(self.defense_names)
        self.M = list(range(self.m))

        # Установка параметров из наборов данных
        self.u = np.array(attack_data['damage_potential'])
        self.c_attack = np.array(attack_data['cost'])
        self.c_defense = np.array(defense_data['cost'])
        self.w_max = np.array(attack_data['max_investment'])
        self.z_max = np.array(defense_data['max_investment'])

        # Загрузка матрицы эффективности
        self.P = load_effectiveness_matrix(effectiveness_dataset, self.attack_names, self.defense_names)

        # Установка бюджетных ограничений на основе значений из набора данных
        self.C_max_defense = np.sum(self.c_defense) * 0.6  # 60% от общей возможной стоимости
        self.C_max_attack = np.sum(self.c_attack) * 0.7  # 70% от общей возможной стоимости

        # Коэффициенты эффективности инвестиций - могут быть получены из исторических данных
        # Пока используем случайные значения
        self.alpha_defense = np.random.uniform(0.001, 0.005, self.m)
        self.alpha_attack = np.random.uniform(0.001, 0.003, self.n)

        # Временной период
        self.T = 1.0  # Нормализованный временной период

        # Вывод ключевых параметров
        print(f"Загружено {self.n} типов атак и {self.m} механизмов защиты из наборов данных")
        print(f"Бюджет защиты: ${self.C_max_defense:.2f}")
        print(f"Бюджет атаки: ${self.C_max_attack:.2f}")

    # Существующие методы из оригинального класса
    def initialize_parameters(self):
        """Initialize all game parameters with realistic values for DDoS scenario."""
        # Potential damage from each attack type (in monetary value or service disruption score)
        self.u = np.random.uniform(10000, 100000, self.n)  # Damage values for each attack

        # Attack costs - different types of DDoS attacks have different implementation costs
        self.c_attack = np.random.uniform(1000, 10000, self.n)

        # Defense costs - firewalls, rate limiters, traffic analyzers, etc.
        self.c_defense = np.random.uniform(5000, 50000, self.m)

        # Maximum investment limits
        self.w_max = np.random.uniform(1000, 5000, self.n)
        self.z_max = np.random.uniform(5000, 20000, self.m)

        # Budget constraints
        self.C_max_defense = np.sum(self.c_defense) * 0.6  # 60% of total possible cost
        self.C_max_attack = np.sum(self.c_attack) * 0.7  # 70% of total possible cost

        # Base effectiveness of defenses against attacks (probability matrix)
        self.P = np.random.uniform(0.1, 0.9, (self.n, self.m))

        # Investment efficiency coefficients
        self.alpha_defense = np.random.uniform(0.001, 0.005, self.m)  # Diminishing returns
        self.alpha_attack = np.random.uniform(0.001, 0.003, self.n)  # Diminishing returns

        # Time period
        self.T = 1.0  # Normalized time period

        # Print key parameters
        print(f"Бюджет защиты: ${self.C_max_defense:.2f}")
        print(f"Бюджет атаки: ${self.C_max_attack:.2f}")


    def calculate_defense_effectiveness(self, z: np.ndarray) -> np.ndarray:
        """Calculate defense effectiveness boost from investments."""
        effectiveness = np.zeros(self.m)
        for j in range(self.m):
            effectiveness[j] = 1.0 - np.exp(-self.alpha_defense[j] * z[j])
        return effectiveness

    def calculate_attack_effectiveness(self, w: np.ndarray) -> np.ndarray:
        """Calculate attack effectiveness boost from investments."""
        effectiveness = np.zeros(self.n)
        for i in range(self.n):
            effectiveness[i] = 1.0 - np.exp(-self.alpha_attack[i] * w[i])
        return effectiveness


def run_model_with_real_data(dataset_dir="./datasets", download=True, use_cicddos=False):
    """
    Запуск модели защиты от DDoS с реальными данными

    Args:
        dataset_dir: Директория для хранения наборов данных
        download: Загружать ли набор данных, если его нет
        use_cicddos: Использовать ли набор данных CIC-DDoS2019
    """
    import os

    # Создание директории, если она не существует
    os.makedirs(dataset_dir, exist_ok=True)

    # Пути для обработанных наборов данных
    attack_data_path = os.path.join(dataset_dir, "processed_attacks.csv")
    defense_data_path = os.path.join(dataset_dir, "defense_mechanisms.csv")
    effectiveness_path = os.path.join(dataset_dir, "effectiveness_matrix.csv")

    # Проверка, нужно ли обработать данные
    if download or not (os.path.exists(attack_data_path) and
                        os.path.exists(defense_data_path) and
                        os.path.exists(effectiveness_path)):

        if use_cicddos:
            # Директория с файлами CIC-DDoS2019
            cicddos_dir = os.path.join(dataset_dir, "CIC-DDoS-2019")
            if not os.path.exists(cicddos_dir) or not os.listdir(cicddos_dir):
                print("Директория с набором данных CIC-DDoS2019 не найдена или пуста.")
                print(f"Пожалуйста, скопируйте CSV-файлы в директорию: {cicddos_dir}")
                return None

            process_cicddos2019_dataset(cicddos_dir, attack_data_path)
        else:
            # Создание данных атак на основе распространенных типов DDoS
            attack_data = {
                'attack_name': ['UDP Flood', 'TCP SYN Flood', 'HTTP Flood', 'DNS Amplification',
                                'NTP Amplification', 'LDAP', 'MSSQL', 'NetBIOS', 'SNMP', 'SSDP',
                                'WebDDoS', 'TFTP'],
                'estimated_damage': [50000, 45000, 60000, 80000, 70000, 65000, 55000,
                                     40000, 42000, 58000, 75000, 38000],
                'execution_cost': [2000, 3000, 5000, 4000, 3500, 4500, 6000,
                                   2800, 3200, 3800, 5500, 2500],
                'max_investment': [4000, 6000, 10000, 8000, 7000, 9000, 12000,
                                   5600, 6400, 7600, 11000, 5000]
            }

            pd.DataFrame(attack_data).to_csv(attack_data_path, index=False)
            print(f"Создан синтетический набор данных атак с {len(attack_data['attack_name'])} типами")

        # Создание данных о механизмах защиты
        defense_names = create_defense_dataset(defense_data_path)

        # Создание матрицы эффективности
        if use_cicddos:
            # Загрузка имен атак из обработанного набора данных
            attack_df = pd.read_csv(attack_data_path)
            attack_names = attack_df['attack_name'].tolist()
        else:
            attack_names = attack_data['attack_name']

        create_effectiveness_matrix(effectiveness_path, attack_names, defense_names)

    # Проверка наличия всех необходимых файлов
    if not (os.path.exists(attack_data_path) and os.path.exists(defense_data_path) and os.path.exists(
            effectiveness_path)):
        print("Ошибка: не удалось создать или найти все необходимые файлы данных.")
        return None

    # Инициализация параметров игры из наборов данных
    params = GameParameters(
        attack_dataset=attack_data_path,
        defense_dataset=defense_data_path,
        effectiveness_dataset=effectiveness_path,
        seed=42
    )

    return params


if __name__ == "__main__":
    import os
    import time
    from datetime import datetime
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd
    from ddos_game_solver import DiscreteContGameSolver

    # Настройка пути и параметров
    dataset_dir = "./datasets"
    results_dir = "./results"
    os.makedirs(results_dir, exist_ok=True)

    # Запуск с метками времени для логирования
    start_time = time.time()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print(f"=== Запуск оптимизации защиты от DDoS-атак ({timestamp}) ===")
    print(f"Настройка и инициализация параметров из данных CIC-DDoS2019...")

    # Инициализация параметров из реальных данных
    params = run_model_with_real_data(
        dataset_dir=dataset_dir,
        download=True,
        use_cicddos=True  # Установите True для использования CIC-DDoS2019
    )

    # Проверка успешности инициализации параметров
    if params is None:
        print("Ошибка: Не удалось инициализировать параметры из данных. Завершение работы.")
        exit(1)

    # Вывод основных параметров для подтверждения
    print("\nОбзор инициализированных параметров:")
    print(f"Количество типов атак: {params.n}")
    print(f"Количество механизмов защиты: {params.m}")
    print(f"Бюджет защиты: ${params.C_max_defense:.2f}")
    print(f"Бюджет атаки: ${params.C_max_attack:.2f}")

    # Настройка и инициализация решателя
    print("\nИнициализация решателя дискретно-непрерывной игры...")

    # Параметры генетического алгоритма
    pop_size = max(50, params.n * 5)  # Размер популяции зависит от числа атак
    max_gen = 150  # Максимальное число поколений
    crossover_prob = 0.8  # Вероятность скрещивания
    mutation_prob = 0.2  # Вероятность мутации
    tournament_size = 3  # Размер турнира для селекции

    solver = DiscreteContGameSolver(
        params=params,
        pop_size=pop_size,
        max_gen=max_gen,
        crossover_prob=crossover_prob,
        mutation_prob=mutation_prob,
        tournament_size=tournament_size
    )

    # Запуск коэволюционного процесса
    print("\nЗапуск процесса коэволюции для поиска оптимальных стратегий...")
    best_defense, best_attack = solver.evolve()

    # Проверка равновесия
    print("\nПроверка равновесия Нэша для найденных стратегий...")
    is_nash = solver.verify_nash_equilibrium(best_defense, best_attack)

    # Анализ и визуализация результатов
    print("\nАнализ оптимальных стратегий...")
    results = solver.analyze_results(best_defense, best_attack)

    # Вывод результатов о выбранных стратегиях
    print("\n=== Результаты оптимизации защиты от DDoS-атак ===")

    print("\nОптимальная стратегия защиты:")
    if hasattr(params, 'defense_names'):
        for j, defense_name in enumerate(results['defense_names']):
            defense_idx = results['defense_choices'][j]
            base_cost = results['defense_base_costs'][j]
            investment = results['defense_investments'][j]
            print(f"  {defense_name}:")
            print(f"    - Базовая стоимость: ${base_cost:.2f}")
            print(f"    - Инвестиции: ${investment:.2f}")
            print(f"    - Общая стоимость: ${base_cost + investment:.2f}")
    else:
        for j, idx in enumerate(results['defense_choices']):
            print(f"  Защита {idx}:")
            print(f"    - Базовая стоимость: ${results['defense_base_costs'][j]:.2f}")
            print(f"    - Инвестиции: ${results['defense_investments'][j]:.2f}")
            print(f"    - Общая стоимость: ${results['defense_base_costs'][j] + results['defense_investments'][j]:.2f}")

    print(f"\nИспользование бюджета защиты: ${np.sum(results['defense_total_costs']):.2f} " +
          f"из ${params.C_max_defense:.2f} ({results['defense_budget_usage'] * 100:.1f}%)")

    print("\nОптимальная стратегия атаки:")
    if hasattr(params, 'attack_names'):
        for i, attack_name in enumerate(results['attack_names']):
            attack_idx = results['attack_choices'][i]
            base_cost = results['attack_base_costs'][i]
            investment = results['attack_investments'][i]
            damage = results['attack_damages'][i]
            print(f"  {attack_name}:")
            print(f"    - Потенциальный ущерб: ${damage:.2f}")
            print(f"    - Базовая стоимость: ${base_cost:.2f}")
            print(f"    - Инвестиции: ${investment:.2f}")
    else:
        for i, idx in enumerate(results['attack_choices']):
            print(f"  Атака {idx}:")
            print(f"    - Потенциальный ущерб: ${results['attack_damages'][i]:.2f}")
            print(f"    - Базовая стоимость: ${results['attack_base_costs'][i]:.2f}")
            print(f"    - Инвестиции: ${results['attack_investments'][i]:.2f}")

    print(f"\nИспользование бюджета атаки: ${np.sum(results['attack_total_costs']):.2f} " +
          f"из ${params.C_max_attack:.2f} ({results['attack_budget_usage'] * 100:.1f}%)")

    print(f"\nОжидаемый ущерб: ${results['damage_value']:.2f}")

    # Анализ покрытия атак
    print("\nАнализ покрытия атак:")
    attack_coverage = results['attack_coverage']

    if len(results['unprotected_attacks']) > 0:
        print("Слабо защищенные атаки (вероятность предотвращения < 20%):")
        for idx in results['unprotected_attacks']:
            attack_name = results['attack_names'][list(results['attack_choices']).index(idx)] if hasattr(params,
                                                                                                         'attack_names') else f"Атака {idx}"
            print(f"  - {attack_name}: {attack_coverage[idx] * 100:.1f}% покрытия")
    else:
        print("Все выбранные атаки имеют базовое покрытие защитой.")

    if len(results['well_protected_attacks']) > 0:
        print("Хорошо защищенные атаки (вероятность предотвращения > 80%):")
        for idx in results['well_protected_attacks']:
            attack_name = results['attack_names'][list(results['attack_choices']).index(idx)] if hasattr(params,
                                                                                                         'attack_names') else f"Атака {idx}"
            print(f"  - {attack_name}: {attack_coverage[idx] * 100:.1f}% покрытия")

    # Визуализация результатов
    print("\nВизуализация результатов...")
    solver.visualize_results(results)

    # Сохранение результатов
    result_file = os.path.join(results_dir, f"ddos_defense_results_{timestamp}.csv")

    # Создание DataFrame с результатами
    results_df = pd.DataFrame({
        'Тип данных': ['CIC-DDoS2019' if hasattr(params, 'attack_names') else 'Синтетические'],
        'Количество атак': [params.n],
        'Количество защит': [params.m],
        'Выбрано атак': [len(results['attack_choices'])],
        'Выбрано защит': [len(results['defense_choices'])],
        'Бюджет защиты': [params.C_max_defense],
        'Использовано бюджета защиты': [np.sum(results['defense_total_costs'])],
        'Бюджет атаки': [params.C_max_attack],
        'Использовано бюджета атаки': [np.sum(results['attack_total_costs'])],
        'Ожидаемый ущерб': [results['damage_value']],
        'Равновесие Нэша': [is_nash],
        'Время выполнения (сек)': [time.time() - start_time]
    })

    # Сохранение основных результатов
    results_df.to_csv(result_file)
    print(f"Результаты сохранены в файл: {result_file}")

    # Сохранение графиков
    fig_file = os.path.join(results_dir, f"ddos_defense_visualization_{timestamp}.png")
    plt.savefig(fig_file)
    print(f"Визуализация сохранена в файл: {fig_file}")

    print(f"\nВремя выполнения: {time.time() - start_time:.2f} секунд")
    print("Программа успешно завершена.")
