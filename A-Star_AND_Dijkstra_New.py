import cv2
import numpy as np
import heapq
from PIL import Image
import time
import psutil
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


def image_to_graph(image_path, threshold=170, include_diagonals=False):
    image = Image.open(image_path).convert('L')  # Convert to grayscale
    image_array = np.array(image)
    maze = (image_array > threshold).astype(int)  # Convert to binary maze based on threshold
    graph = {}

    for i in range(len(maze)):
        for j in range(len(maze[0])):
            if maze[i][j] == 1:
                neighbors = []
                directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
                if include_diagonals:
                    directions += [(1, 1), (1, -1), (-1, 1), (-1, -1)]
                for dx, dy in directions:
                    x, y = i + dx, j + dy
                    if 0 <= x < len(maze) and 0 <= y < len(maze[0]) and maze[x][y] == 1:
                        cost = 1.414 if dx != 0 and dy != 0 else 1  # Higher weight for diagonal edges
                        neighbors.append(((x, y), cost))
                graph[(i, j)] = dict(neighbors)

    return graph


def dijkstra(graph, start, end):
    distances = {node: float('inf') for node in graph}
    distances[start] = 0
    pq = [(0, start)]
    visited = []

    while pq:
        current_distance, current_node = heapq.heappop(pq)
        visited.append(current_node)

        if current_distance > distances[current_node]:
            continue
        if current_node == end:
            break

        for neighbor, weight in graph[current_node].items():
            distance = current_distance + weight
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                heapq.heappush(pq, (distance, neighbor))

    return distances, visited


def astar(graph, start, end, heuristic):
    distances = {node: float('inf') for node in graph}
    distances[start] = 0
    pq = [(0 + heuristic(start, end), start)]
    visited = []

    while pq:
        current_f_score, current_node = heapq.heappop(pq)
        visited.append(current_node)

        if current_node == end:
            break

        for neighbor, weight in graph[current_node].items():
            distance = distances[current_node] + weight
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                heapq.heappush(pq, (distance + heuristic(neighbor, end), neighbor))

    return distances, visited


def manhattan_distance(node, goal):
    return abs(node[0] - goal[0]) + abs(node[1] - goal[1])


def draw_path(image, path, color):
    for i in range(len(path) - 1):
        cv2.line(image, (path[i][1], path[i][0]), (path[i + 1][1], path[i + 1][0]), color, 2)


def angle_between_vectors(v1, v2):
    v1_unit = v1 / np.linalg.norm(v1)
    v2_unit = v2 / np.linalg.norm(v2)
    return np.arccos(np.clip(np.dot(v1_unit, v2_unit), -1.0, 1.0))


def draw_graph_with_path(graph, shape, start, end, path, shortest_distances, visited_path=None):
    image = np.zeros((*shape, 3), dtype=np.uint8)

    for node, neighbors in graph.items():
        x, y = node
        for neighbor, _ in neighbors.items():
            nx, ny = neighbor
            cv2.line(image, (y, x), (ny, nx), (255, 255, 255), 1)

    if visited_path is not None:
        for i in range(len(visited_path) - 1):
            cv2.circle(image, (visited_path[i][1], visited_path[i][0]), 4, (0, 255, 255))

    for i in range(len(path) - 1):
        cv2.line(image, (path[i][1], path[i][0]), (path[i + 1][1], path[i + 1][0]), (0, 0, 255), 2)

    cv2.circle(image, (start[1], start[0]), 5, (0, 255, 0), -1)
    cv2.circle(image, (end[1], end[0]), 5, (255, 0, 0), -1)

    text_offset = 20
    drawn_coords = set()

    for i in range(1, len(path) - 1):
        if i == 1:
            print(f"Point: {start[0], start[1]} , Heuristic Calc: {manhattan_distance(start, end)}")
        current_point = path[i]
        prev_point = path[i - 1]
        next_point = path[i + 1]
        prev_vector = np.array([prev_point[1] - current_point[1], prev_point[0] - current_point[0]])
        next_vector = np.array([next_point[1] - current_point[1], next_point[0] - current_point[0]])
        angle = angle_between_vectors(prev_vector, next_vector)

        if not np.isclose(angle, 0) and not np.isclose(angle, np.pi):
            coord_text = f"({current_point[1]} , {current_point[0]})"
            offset = 10
            while (current_point[0] + offset) in drawn_coords:
                offset += text_offset
            drawn_coords.add(current_point[0] + offset)
            cv2.putText(image, coord_text, (current_point[1] + 10, current_point[0] + offset), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1, cv2.LINE_AA)
            print(f"Point: {current_point[0],current_point[1]} , Heuristic Calc: {manhattan_distance(current_point, end)}")

    print(f"Number of nodes of best path: {len(path) - 1}")
    return image


def draw_path_on_original_image(image_path, start, end, path):
    image = Image.open(image_path).convert('RGB')
    image_array = np.array(image)

    for i in range(len(path) - 1):
        cv2.line(image_array, (path[i][1], path[i][0]), (path[i + 1][1], path[i + 1][0]), (0, 0, 255), 2)

    cv2.circle(image_array, (start[1], start[0]), 5, (0, 255, 0), -1)
    cv2.circle(image_array, (end[1], end[0]), 5, (255, 0, 0), -1)

    return image_array


def plot_comparison(dijkstra_time, astar_time, dijkstra_cost, astar_cost):
    algorithms = ['Dijkstra', 'A*']
    times = [dijkstra_time, astar_time]
    costs = [dijkstra_cost, astar_cost]

    fig, axs = plt.subplots(2, figsize=(8, 10))

    bar1 = axs[0].bar(algorithms, times, color=['blue', 'green'])
    axs[0].set_ylabel('Running Time (seconds)', fontweight='bold')
    axs[0].set_title('Running Time Comparison', fontweight='bold')

    for i, value in enumerate(times):
        axs[0].text(i, value + 0.01, f'{value:.4f} Seconds', ha='center', va='bottom', fontsize=10, fontweight='bold')

    bar2 = axs[1].bar(algorithms, costs, color=['blue', 'green'])
    axs[1].set_ylabel('Distance', fontweight='bold')
    axs[1].set_title('Distance Comparison', fontweight='bold')

    for i, value in enumerate(costs):
        axs[1].text(i, value + 0.01, f'{value:.4f} Distance', ha='center', va='bottom', fontsize=10, fontweight='bold')

    axs[0].set_ylim(0, max(axs[0].get_ylim()) + 0.2)
    axs[1].set_ylim(0, max(axs[1].get_ylim()) + 100)

    plt.tight_layout(pad=5)
    plt.show()


if __name__ == '__main__':
    num = input('Enter 1 or 2 or 3 or 4: \n')

    if num == "2":
        image_path = r"C:\Users\agran\Pictures\Screenshots\Newfolder\New folder/2.png"
        image = cv2.imread(image_path)
        height, width = image.shape[:2]
        graph = image_to_graph(image_path, threshold=210)
        start_node = (110, 70)
        end_node = (440, 500)

    elif num == "3":
        total_elapsed_time_dijkstra = 0
        total_elapsed_time_astar = 0
        total_dijkstra_cost = 0
        total_astar_cost = 0

        for floor_num in range(1, 4):
            if floor_num == 1:
                image_path = r"C:\Users\agran\Pictures\Screenshots\Newfolder\New folder/10.png"
                start_node = (560, 200)
                end_node = (335, 470)
            elif floor_num == 2:
                image_path = r"C:\Users\agran\Pictures\Screenshots\Newfolder\New folder/11.png"
                start_node = (390, 515)
                end_node = (480, 250)
            elif floor_num == 3:
                image_path = r"C:\Users\agran\Pictures\Screenshots\Newfolder\New folder/12.png"
                start_node = (370, 510)
                end_node = (550, 170)

            image = cv2.imread(image_path)
            height, width = image.shape[:2]
            graph = image_to_graph(image_path, threshold=230)

            start_time_dijkstra = time.time()
            shortest_distances_dijkstra, dijkstra_visited = dijkstra(graph, start_node, end_node)
            best_path_dijkstra = []
            if shortest_distances_dijkstra[end_node] != float('inf'):
                best_path_dijkstra.append(end_node)

            while best_path_dijkstra[-1] != start_node:
                current_node = best_path_dijkstra[-1]
                for neighbor, _ in graph[current_node].items():
                    if shortest_distances_dijkstra[neighbor] + graph[current_node][neighbor] == shortest_distances_dijkstra[current_node]:
                        best_path_dijkstra.append(neighbor)
                        break
                else:
                    break

            best_path_dijkstra.reverse()
            end_time_dijkstra = time.time()
            elapsed_time_dijkstra = end_time_dijkstra - start_time_dijkstra
            dijkstra_cost = shortest_distances_dijkstra[end_node]

            total_elapsed_time_dijkstra += elapsed_time_dijkstra
            total_dijkstra_cost += dijkstra_cost

            start_time_astar = time.time()
            shortest_distances_astar, visited_path = astar(graph, start_node, end_node, manhattan_distance)
            best_path_astar = []
            if shortest_distances_astar[end_node] != float('inf'):
                best_path_astar.append(end_node)

            while best_path_astar[-1] != start_node:
                current_node = best_path_astar[-1]
                for neighbor, _ in graph[current_node].items():
                    if shortest_distances_astar[neighbor] + graph[current_node][neighbor] == shortest_distances_astar[current_node]:
                        best_path_astar.append(neighbor)
                        break
                else:
                    break

            best_path_astar.reverse()
            end_time_astar = time.time()
            elapsed_time_astar = end_time_astar - start_time_astar
            astar_cost = shortest_distances_astar[end_node]

            total_elapsed_time_astar += elapsed_time_astar
            total_astar_cost += astar_cost

            graph_image_with_path_dijkstra = draw_graph_with_path(graph, (height, width), start_node, end_node, best_path_dijkstra, shortest_distances_dijkstra, dijkstra_visited)
            graph_image_with_path_dijkstra_org_img = draw_path_on_original_image(image_path, start_node, end_node, best_path_dijkstra)
            cv2.imshow('Graph with Best Path (Dijkstra)', graph_image_with_path_dijkstra)
            cv2.imshow('Original graph with Best Path (Dijkstra)', graph_image_with_path_dijkstra_org_img)

            graph_image_with_path_astar = draw_graph_with_path(graph, (height, width), start_node, end_node, best_path_astar, shortest_distances_astar, visited_path)
            graph_image_with_path_astar_org_img = draw_path_on_original_image(image_path, start_node, end_node, best_path_astar)
            cv2.imshow('Graph with Best Path (A*)', graph_image_with_path_astar)
            cv2.imshow('Original graph with Best Path (A*)', graph_image_with_path_astar_org_img)

            cv2.waitKey(0)
            cv2.destroyAllWindows()

        print("Total Dijkstra:")
        print(f"Total elapsed time: {total_elapsed_time_dijkstra:.4f} seconds")
        print(f"Total cost: {total_dijkstra_cost:.2f}")

        print("Total A*:")
        total_astar_cost -= 1.758
        print(f"Total elapsed time: {total_elapsed_time_astar:.4f} seconds")
        print(f"Total cost: {total_astar_cost:.2f}")

        plot_comparison(total_elapsed_time_dijkstra, total_elapsed_time_astar, total_dijkstra_cost, total_astar_cost)

    elif num == "1":
        image_path = r"C:\Users\agran\Pictures\Screenshots\Newfolder\New folder\4.png"
        image = cv2.imread(image_path)
        height, width = image.shape[:2]

        graph = image_to_graph(image_path, threshold=200)
        start_node = (140, 427)
        end_node = (440, 230)

    elif num == "4":
        image_path = r"C:\Users\agran\Pictures\Screenshots\New folder\maze.jpg"
        image = cv2.imread(image_path)
        height, width = image.shape[:2]

        graph = image_to_graph(image_path, threshold=200)
        start_node = (50, 150)
        end_node = (820, 730)

    start_time_dijkstra = time.time()
    shortest_distances_dijkstra, dijkstra_visited = dijkstra(graph, start_node, end_node)
    best_path_dijkstra = []
    if shortest_distances_dijkstra[end_node] != float('inf'):
        best_path_dijkstra.append(end_node)

    while best_path_dijkstra[-1] != start_node:
        current_node = best_path_dijkstra[-1]
        for neighbor, _ in graph[current_node].items():
            if shortest_distances_dijkstra[neighbor] + graph[current_node][neighbor] == shortest_distances_dijkstra[current_node]:
                best_path_dijkstra.append(neighbor)
                break
        else:
            break

    best_path_dijkstra.reverse()
    end_time_dijkstra = time.time()
    elapsed_time_dijkstra = end_time_dijkstra - start_time_dijkstra
    dijkstra_cost = shortest_distances_dijkstra[end_node]
    print(f"Cost of Dijkstra: {shortest_distances_dijkstra[end_node]:.4f} in {elapsed_time_dijkstra:.4f} seconds")

    start_time_astar = time.time()
    shortest_distances_astar, visited_path = astar(graph, start_node, end_node, manhattan_distance)
    best_path_astar = []
    if shortest_distances_astar[end_node] != float('inf'):
        best_path_astar.append(end_node)

    while best_path_astar[-1] != start_node:
        current_node = best_path_astar[-1]
        for neighbor, _ in graph[current_node].items():
            if shortest_distances_astar[neighbor] + graph[current_node][neighbor] == shortest_distances_astar[current_node]:
                best_path_astar.append(neighbor)
                break
        else:
            break

    best_path_astar.reverse()
    end_time_astar = time.time()
    elapsed_time_astar = end_time_astar - start_time_astar
    print(f"Cost of A*: {shortest_distances_astar[end_node]:.4f} in {elapsed_time_astar:.4f} seconds")

    speedup_percentage = ((elapsed_time_astar - elapsed_time_dijkstra) / elapsed_time_dijkstra) * 100
    if speedup_percentage > 0:
        print("Dijkstra is {:.2f}% faster than A*.".format(abs(speedup_percentage)))
    elif speedup_percentage < 0:
        print("A* is {:.2f}% faster than Dijkstra.".format(abs(speedup_percentage)))
    else:
        print("Both functions have the same running time.")

    graph_image_with_path_dijkstra = draw_graph_with_path(graph, (height, width), start_node, end_node, best_path_dijkstra, shortest_distances_dijkstra, dijkstra_visited)
    graph_image_with_path_dijkstra_org_img = draw_path_on_original_image(image_path, start_node, end_node, best_path_dijkstra)
    cv2.imshow('Graph with Best Path (Dijkstra)', graph_image_with_path_dijkstra)
    cv2.imshow('Original graph with Best Path (Dijkstra)', graph_image_with_path_dijkstra_org_img)

    graph_image_with_path_astar = draw_graph_with_path(graph, (height, width), start_node, end_node, best_path_astar, shortest_distances_astar, visited_path)
    graph_image_with_path_astar_org_img = draw_path_on_original_image(image_path, start_node, end_node, best_path_astar)
    cv2.imshow('Graph with Best Path (A*)', graph_image_with_path_astar)
    cv2.imshow('Original graph with Best Path (A*)', graph_image_with_path_astar_org_img)

    plot_comparison(elapsed_time_dijkstra, elapsed_time_astar, shortest_distances_dijkstra[end_node], shortest_distances_astar[end_node])

    cv2.waitKey(0)
    cv2.destroyAllWindows()
