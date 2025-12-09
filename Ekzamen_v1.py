#Реализация на языке Python
# Концепция решения задачи
# Для решения задачи используем алгоритм Дейкстра, на вход подаем неориентированного граф,
# где города — это вершины, а дороги — рёбра с тремя весами: длина, время и стоимость.
# Парсинг входных данных реализован посекционно: сначала читаем все строки,
# затем разделяем их на секции [CITIES], [ROADS] и [REQUESTS].
# Города сохраняем в два словаря — название: ID и ID: название, чтобы быстро находить как по названию, так и по номеру.
# Дороги добавляем в список смежности сразу в обе стороны, так как граф неориентированный.
# Для поиска оптимального маршрута по каждому критерию отдельно выбираем алгоритм Дейкстры.
# Сложность O((V + E) log V), где V - число городов, E - число дорог.Эффективно даже при сотнях городов.
# Запускаем Дейкстру трижды для каждого запроса:
# первый раз — с весом «длина»,
# второй — «время»,
# третий — «стоимость».
# Каждый раз получаем маршрут, оптимальный по одному из критериев.
# После этого подсчитываем все три метрики (длину, время, стоимость) для каждого из этих маршрутов с помощью обхода пути
# и суммирования параметров по рёбрам
# Компромиссный маршрут выбираем не заново, а среди уже найденных трёх.
# Выбор делаем по заданным приоритетам через  лексикографическое сравнение, реализованное через цикл по индексам приоритетов.

import heapq
from typing import List, Tuple, Dict


def dijkstra(
        graph: Dict[int, List[Tuple[int, int, int, int]]],
        start: int,
        end: int,
        weight_index: int  # 0=длина, 1=время, 2=стоимость
) -> Tuple[int, List[int]]:
    """
    Реализация алгоритма Дейкстры:
    graph — неориентированный граф в виде списка смежности:
    Ключ: ID города (int)
    Значение: список кортежей (сосед, длина, время, стоимость)
    start, end — ID начального и конечного городов.
    weight_index — индекс критерия: 0 → длина, 1 → время, 2 → стоимость.
    возвращает (мин_значение, путь_как_список_id)
    если путь не найден — (inf, [])
    """
    dist = {} # задаем словарь: город → минимальное значение критерия от start до него.
    prev = {} #  задаем словарь: город → предыдущий город на кратчайшем пути (для восстановления маршрута).
    pq = [(0, start)] # приоритетная очередь
    dist[start] = 0 # инициализируем расстояние до start = 0.

    while pq:
        d, u = heapq.heappop(pq) # Извлекаем вершину u с минимальным d.
        if d != dist.get(u, float('inf')):
            continue # ленивая очередь - если d устарело (уже найден более короткий путь), пропускаем.
        if u == end:
            break # как только достигли цели — завершаем поиск (оптимизация алгоритма).
        for v, length, time, cost in graph.get(u, []): # Обходим всех соседей v вершины u.
            weights = [length, time, cost]
            w = weights[weight_index] # выбираем вес w по индексу (0, 1 или 2).
            new_dist = d + w # вычисляем новое расстояние
            if new_dist < dist.get(v, float('inf')): # Если найден более короткий путь до v, обновляем на dist[v] — новое значение, и ставим в приоритетную очередь
                dist[v] = new_dist
                prev[v] = u
                heapq.heappush(pq, (new_dist, v))

    if end not in dist:
        return float('inf'), [] # если end недостижим — возвращаем "inf" и пустой путь.

    # Восстановление пути
    path = []
    cur = end
    while cur != start:
        path.append(cur)
        cur = prev[cur] # Идём от end к start через prev.
    path.append(start) # добавляем start в конец списка.
    path.reverse() # разворачиваем, чтобы получить порядок start → ... → end.
    return dist[end], path


def total_metrics(
        graph: Dict[int, List[Tuple[int, int, int, int]]],
        path: List[int]
) -> Tuple[int, int, int]:
    """Вычисляет суммарные значения всех трёх критериев Д, В, С для заданного маршрута."""
    if len(path) < 2:
        return 0, 0, 0 # пустой или одногородный путь, все метрики = 0.
    total_len = total_time = total_cost = 0
    for i in range(len(path) - 1): # проходим по каждой паре соседних городов в маршруте.
        u, v = path[i], path[i + 1]
        found = False
        for w, l, t, c in graph[u]:
            if w == v:  # ищем в graph[u] ребро к v.
                total_len += l
                total_time += t
                total_cost += c
                found = True
                break
        if not found:
            return float('inf'), float('inf'), float('inf') # если не найдено — возвращаем "inf".
    return total_len, total_time, total_cost # возвращаем кортеж (Д, В, С).


def parse_input(filename: str):
    """Читаем и парсим входной файл input.txt, возвращаем структуры данных для обработки."""
    with open(filename, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]
    # разделение на секции
    sections = {}
    current_section = None
    for line in lines:
        if line.startswith('[') and line.endswith(']'):
            current_section = line[1:-1]
            sections[current_section] = []
        else:
            if current_section:
                sections[current_section].append(line)

    # Парсим города
    city_name_to_id = {}
    city_id_to_name = {}
    for line in sections.get('CITIES', []):
        if ':' in line:
            cid, name = line.split(':', 1)
            cid = int(cid.strip())
            name = name.strip()
            city_name_to_id[name] = cid #1 словарь city_name_to_id: 'Москва' → 1
            city_id_to_name[cid] = name #2 словарь city_id_to_name: 1 → 'Москва'

    # Парсим дороги
    graph = {}
    for line in sections.get('ROADS', []):
        if ':' in line:
            nodes_part, metrics_part = line.split(':', 1)
            nodes_part = nodes_part.replace('-', ' ').strip()
            ids = list(map(int, nodes_part.split()))
            if len(ids) != 2:
                continue
            u, v = ids
            metrics = list(map(int, metrics_part.replace(',', ' ').split()))
            if len(metrics) != 3:
                continue
            length, time, cost = metrics

            if u not in graph:
                graph[u] = []
            if v not in graph:
                graph[v] = []

            graph[u].append((v, length, time, cost))
            graph[v].append((u, length, time, cost)) #добавляет ребро в обе стороны (неориентированный граф).

    # Парсим запросы
    requests = []
    for line in sections.get('REQUESTS', []):
        if '->' in line and '|' in line:
            route_part, prio_part = line.split('|', 1)
            route_part = route_part.strip()
            from_city, to_city = map(str.strip, route_part.split('->'))
            prio_part = prio_part.strip()
            if prio_part.startswith('(') and prio_part.endswith(')'):
                prio_str = prio_part[1:-1]
                priorities = [p.strip() for p in prio_str.split(',')]
            else:
                priorities = ['Д', 'В', 'С']
            requests.append((from_city, to_city, priorities))

    return city_name_to_id, city_id_to_name, graph, requests


def format_route(city_id_to_name: Dict[int, str], path: List[int]) -> str:
    """Преобразует [1, 2, 11] → 'Москва -> Санкт-Петербург -> Уфа'.
        если город не найден — выводит ID (защита от ошибок)."""
    return ' -> '.join(city_id_to_name.get(cid, str(cid)) for cid in path)


def main():
    """Основное тело программы"""
    city_name_to_id, city_id_to_name, graph, requests = parse_input('input.txt')

    CRIT = {
        'Д': ('ДЛИНА', 0),
        'В': ('ВРЕМЯ', 1),
        'С': ('СТОИМОСТЬ', 2)
    }
    # Обработка запроса
    for idx, (from_name, to_name, priorities) in enumerate(requests, start=1):
        output_lines = []
        # Проверка существования города
        if from_name not in city_name_to_id or to_name not in city_name_to_id:
            output_lines.append(f"Error: неизвестный город в запросе '{from_name} -> {to_name}'")
        else:
            start_id = city_name_to_id[from_name]
            end_id = city_name_to_id[to_name]

            # Получаем три маршрута
            routes = {}
            for crit_key in ['Д', 'В', 'С']: # Поиск маршрутов
                _, weight_idx = CRIT[crit_key]
                _, path = dijkstra(graph, start_id, end_id, weight_idx) # запускаем Дейкстру по каждому критерию
                if path:
                    metrics = total_metrics(graph, path)
                    routes[crit_key] = (path, metrics)
                else:
                    routes[crit_key] = ([], (float('inf'), float('inf'), float('inf')))

            # Вывод трёх маршрутов
            for crit_key in ['Д', 'В', 'С']:
                path, (d, v, s) = routes[crit_key]
                crit_name, _ = CRIT[crit_key]
                if path:
                    route_str = format_route(city_id_to_name, path)
                    output_lines.append(f"{crit_name}: {route_str} | Д={d}, В={v}, С={s}")
                else:
                    output_lines.append(f"{crit_name}: НЕТ МАРШРУТА")

            # Выбор компромиссного маршрута
            candidate_routes = {}
            for crit_key in ['Д', 'В', 'С']:
                path, metrics = routes[crit_key]
                if path:
                    candidate_routes[tuple(path)] = metrics

            if not candidate_routes:
                output_lines.append("КОМПРОМИСС: НЕТ МАРШРУТА")
            else:
                priority_indices = [] # Преобразует (С,Д,В) → [2, 0, 1] — порядок сравнения.
                for p in priorities:
                    if p == 'Д':
                        priority_indices.append(0)
                    elif p == 'В':
                        priority_indices.append(1)
                    elif p == 'С':
                        priority_indices.append(2)

                best_key = None
                best_metrics = None
                # лексикографическое сравнение по приоритетам, как только найдено преимущество, выбираем этот маршрут
                for path_key, metrics in candidate_routes.items():
                    if best_key is None:
                        best_key = path_key
                        best_metrics = metrics
                    else:
                        for idx_p in priority_indices:
                            if metrics[idx_p] < best_metrics[idx_p]:
                                best_key = path_key
                                best_metrics = metrics
                                break
                            elif metrics[idx_p] > best_metrics[idx_p]:
                                break

                best_path = list(best_key)
                route_str = format_route(city_id_to_name, best_path)
                d, v, s = best_metrics
                output_lines.append(f"КОМПРОМИСС: {route_str} | Д={d}, В={v}, С={s}")

        # Запись в отдельный файл
        with open(f'output{idx}.txt', 'w', encoding='utf-8') as f:
            f.write('\n'.join(output_lines))


if __name__ == '__main__': # вход в программу вынесен
    main()


# TEST0
# INPUT
# [CITIES]
# 1: Москва
# 2: Санкт-Петербург
# 3: Нижний Новгород
# 4: Казань
# [ROADS]
# 1 - 2: 700, 480, 800
# 1 - 3: 400, 250, 300
# 2 - 3: 1100, 700, 1200
# 3 - 4: 350, 300, 500
# 1 - 4: 800, 600, 1000
# [REQUESTS]
# Москва -> Санкт-Петербург | (Д,В,С)
# Нижний Новгород -> Казань | (С,В,Д)
# OUTPUT0_1:
# ДЛИНА: Москва -> Санкт-Петербург | Д=700, В=480, С=800
# ВРЕМЯ: Москва -> Санкт-Петербург | Д=700, В=480, С=800
# СТОИМОСТЬ: Москва -> Санкт-Петербург | Д=700, В=480, С=800
# КОМПРОМИСС: Москва -> Санкт-Петербург | Д=700, В=480, С=800
#OUTPUT0_2
# ДЛИНА: Нижний Новгород -> Казань | Д=350, В=300, С=500
# ВРЕМЯ: Нижний Новгород -> Казань | Д=350, В=300, С=500
# СТОИМОСТЬ: Нижний Новгород -> Казань | Д=350, В=300, С=500
# КОМПРОМИСС: Нижний Новгород -> Казань | Д=350, В=300, С=500
# TEST1
# INPUT:
# [CITIES]
# 1: Город A
# 2: Город B
# 3: Город C
# 4: Город D
# 5: Город E
# 6: Город F
# 7: Город G
# 8: Город H
# 9: Город I
# 10: Город J
#
# [ROADS]
# 1 - 2: 10, 15, 20
# 2 - 3: 10, 15, 20
# 3 - 4: 10, 15, 20
# 4 - 5: 10, 15, 20
# 6 - 7: 10, 15, 20
# 7 - 8: 10, 15, 20
# 8 - 9: 10, 15, 20
# 9 - 10: 10, 15, 20
# 1 - 6: 10, 15, 20
# 2 - 7: 10, 15, 20
# 3 - 8: 10, 15, 20
# 4 - 9: 10, 15, 20
# 5 - 10: 10, 15, 20
#
# [REQUESTS]
# Город A -> Город J | (Д,В,С)
# OUTPUT:
# ДЛИНА: Город A -> Город B -> Город C -> Город D -> Город E -> Город J | Д=50, В=75, С=100
# ВРЕМЯ: Город A -> Город B -> Город C -> Город D -> Город E -> Город J | Д=50, В=75, С=100
# СТОИМОСТЬ: Город A -> Город B -> Город C -> Город D -> Город E -> Город J | Д=50, В=75, С=100
# КОМПРОМИСС: Город A -> Город B -> Город C -> Город D -> Город E -> Город J | Д=50, В=75, С=100
# TEST2
# INPUT:
# [CITIES]
# 1: Москва (хаб)
# 2: Санкт-Петербург (хаб)
# 3: Новосибирск (хаб)
# 4: Подмосковье
# 5: Балашиха
# 6: СПб-пригород
# 7: Выборг
# 8: Новосиб-район
# 9: Академгородок
# 10: Курган
#
# [ROADS]
# 1 - 2: 700, 400, 1500
# 1 - 3: 3200, 1800, 4000
# 2 - 3: 4000, 2200, 5000
# 1 - 4: 30, 60, 50
# 1 - 5: 25, 50, 40
# 2 - 6: 40, 80, 60
# 2 - 7: 150, 180, 100
# 3 - 8: 20, 40, 30
# 3 - 9: 10, 25, 20
# 3 - 10: 400, 500, 200
#
# [REQUESTS]
# Подмосковье -> Курган | (С,Д,В)
# OUTPUT:
# ДЛИНА: Подмосковье -> Москва (хаб) -> Новосибирск (хаб) -> Курган | Д=3630, В=2360, С=4250
# ВРЕМЯ: Подмосковье -> Москва (хаб) -> Новосибирск (хаб) -> Курган | Д=3630, В=2360, С=4250
# СТОИМОСТЬ: Подмосковье -> Москва (хаб) -> Новосибирск (хаб) -> Курган | Д=3630, В=2360, С=4250
# КОМПРОМИСС: Подмосковье -> Москва (хаб) -> Новосибирск (хаб) -> Курган | Д=3630, В=2360, С=4250
# TEST3:
# [CITIES]
# 1: Город 1
# 2: Город 2
# 3: Город 3
# 4: Город 4
# 5: Город 5
# 6: Город 6
# 7: Город 7
# 8: Город 8
# 9: Город 9
# 10: Остров
#
# [ROADS]
# 1 - 2: 100, 60, 200
# 2 - 3: 150, 90, 300
# 3 - 4: 200, 120, 400
# 1 - 5: 300, 180, 500
# 5 - 6: 250, 150, 350
# 6 - 7: 100, 70, 150
# 7 - 8: 120, 80, 180
# 8 - 9: 90, 60, 120
#
# [REQUESTS]
# Город 1 -> Город 9 | (В,С,Д)
# Город 1 -> Остров | (Д,В,С)
# OUTPUT3_0:
# ДЛИНА: Город 1 -> Город 5 -> Город 6 -> Город 7 -> Город 8 -> Город 9 | Д=860, В=540, С=1300
# ВРЕМЯ: Город 1 -> Город 5 -> Город 6 -> Город 7 -> Город 8 -> Город 9 | Д=860, В=540, С=1300
# СТОИМОСТЬ: Город 1 -> Город 5 -> Город 6 -> Город 7 -> Город 8 -> Город 9 | Д=860, В=540, С=1300
# КОМПРОМИСС: Город 1 -> Город 5 -> Город 6 -> Город 7 -> Город 8 -> Город 9 | Д=860, В=540, С=1300
#OUTPUT3_1
# ДЛИНА: НЕТ МАРШРУТА
# ВРЕМЯ: НЕТ МАРШРУТА
# СТОИМОСТЬ: НЕТ МАРШРУТА
# КОМПРОМИСС: НЕТ МАРШРУТА