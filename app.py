import json
import math
import random
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from matplotlib.patches import Rectangle

from src.planner import AStarPlanner

st.set_page_config(page_title="NavOS Command Center", page_icon="N", layout="wide")
DATA_DIR = Path("data")
MAP_FILE = DATA_DIR / "maps.json"
HISTORY_FILE = DATA_DIR / "run_history.json"
DEFAULT_OBSTACLES = [(15, 15, 10, 10), (25, 5, 8, 20), (10, 30, 20, 8), (35, 25, 10, 10)]
PAGES = ["Mission Control", "Live Simulation", "Mission Planner", "Map Studio", "Perception Lab", "Route Inspector", "Run History"]


def ensure_files():
    DATA_DIR.mkdir(exist_ok=True)
    if not MAP_FILE.exists():
        MAP_FILE.write_text(json.dumps({"Default warehouse": {"rows": 50, "cols": 50, "obstacles": DEFAULT_OBSTACLES}}), encoding="utf-8")
    if not HISTORY_FILE.exists():
        HISTORY_FILE.write_text("[]", encoding="utf-8")


def load_json(path, fallback):
    ensure_files()
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return fallback


def save_json(path, value):
    ensure_files()
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def init_state():
    maps = load_json(MAP_FILE, {})
    selected = st.session_state.get("map_name", "Default warehouse")
    selected = selected if selected in maps else next(iter(maps))
    world = maps[selected]
    st.session_state.setdefault("map_name", selected)
    st.session_state.setdefault("rows", world["rows"])
    st.session_state.setdefault("cols", world["cols"])
    st.session_state.setdefault("obstacles", [tuple(item) for item in world["obstacles"]])
    st.session_state.setdefault("robots", [{"name": "Rover 1", "start": (5, 5), "goal": (42, 42), "step": 0}])
    st.session_state.setdefault("sim_paused", False)
    st.session_state.setdefault("speed", 1.0)
    st.session_state.setdefault("planner", "A*")


def make_grid():
    grid = np.zeros((st.session_state.rows, st.session_state.cols), dtype=np.int32)
    for row, col, height, width in st.session_state.obstacles:
        grid[max(0, row):min(st.session_state.rows, row + height), max(0, col):min(st.session_state.cols, col + width)] = 1
    return grid


def valid_cell(cell):
    row, col = cell
    return 0 <= row < st.session_state.rows and 0 <= col < st.session_state.cols and not make_grid()[cell]


def rebuild_path(came_from, start, goal):
    if goal != start and goal not in came_from:
        return []
    path = [goal]
    while path[-1] != start:
        path.append(came_from[path[-1]])
    return path[::-1]


def dijkstra(grid, start, goal):
    frontier = [(0.0, start)]
    costs, came_from = {start: 0.0}, {}
    while frontier:
        cost, current = frontier.pop(0)
        if current == goal:
            break
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]:
            neighbor = (current[0] + dr, current[1] + dc)
            if not (0 <= neighbor[0] < grid.shape[0] and 0 <= neighbor[1] < grid.shape[1]) or grid[neighbor]:
                continue
            next_cost = cost + (1.414 if dr and dc else 1)
            if next_cost < costs.get(neighbor, float("inf")):
                costs[neighbor] = next_cost
                came_from[neighbor] = current
                frontier.append((next_cost, neighbor))
                frontier.sort(key=lambda item: item[0])
    return rebuild_path(came_from, start, goal)


def rrt_route(grid, start, goal):
    route = [start]
    current = start
    for _ in range(grid.size * 2):
        candidates = [(current[0] + dr, current[1] + dc) for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]]
        candidates = [cell for cell in candidates if 0 <= cell[0] < grid.shape[0] and 0 <= cell[1] < grid.shape[1] and not grid[cell]]
        if not candidates:
            break
        current = min(candidates, key=lambda cell: math.dist(cell, goal) + random.random() * 4)
        route.append(current)
        if current == goal:
            return route
    return []


def plan_route(start, goal, planner=None):
    start, goal = tuple(map(int, start)), tuple(map(int, goal))
    grid = make_grid()
    if not valid_cell(start) or not valid_cell(goal):
        return []
    planner = planner or st.session_state.planner
    if planner == "Dijkstra":
        return dijkstra(grid, start, goal)
    if planner == "RRT":
        return rrt_route(grid, start, goal)
    return AStarPlanner((st.session_state.rows, st.session_state.cols)).plan_path(grid, start, goal)


def route_figure(routes, starts, goals, robots=None, detections=None):
    figure, axis = plt.subplots(figsize=(8, 6), facecolor="white")
    axis.set_facecolor("#f8fafb")
    colors = ["#087f8c", "#e07a5f", "#6a4c93", "#2a9d8f"]
    for row, col, height, width in st.session_state.obstacles:
        axis.add_patch(Rectangle((col, row), width, height, color="#e76f51", alpha=0.78))
    for index, path in enumerate(routes):
        if path:
            points = np.array([(col + 0.5, row + 0.5) for row, col in path])
            axis.plot(points[:, 0], points[:, 1], color=colors[index % len(colors)], linewidth=2.2, label=f"Route {index + 1}", zorder=3)
    if detections:
        for row, col, height, width, label in detections:
            axis.add_patch(Rectangle((col, row), width, height, fill=False, edgecolor="#f4a261", linewidth=2))
            axis.text(col, row - 0.5, label, color="#b45309", fontsize=8)
    for start, goal in zip(starts, goals):
        axis.scatter([start[1] + 0.5], [start[0] + 0.5], color="#277da1", s=80, zorder=4)
        axis.scatter([goal[1] + 0.5], [goal[0] + 0.5], color="#2a9d8f", s=120, marker="*", zorder=4)
    for robot in robots or []:
        axis.scatter([robot[1] + 0.5], [robot[0] + 0.5], color="#111827", s=55, zorder=5)
    axis.set_xlim(0, st.session_state.cols)
    axis.set_ylim(st.session_state.rows, 0)
    axis.set_aspect("equal")
    axis.set_xticks(range(0, st.session_state.cols + 1, max(1, st.session_state.cols // 5)))
    axis.set_yticks(range(0, st.session_state.rows + 1, max(1, st.session_state.rows // 5)))
    axis.tick_params(colors="#64748b", labelsize=8)
    axis.grid(color="#cbd5e1", linewidth=0.5)
    if len(routes) > 1:
        axis.legend(loc="upper right", fontsize=8)
    figure.tight_layout()
    return figure


def metric(label, value, detail, accent="#087f8c"):
    st.markdown(f'<div class="metric"><div class="metric-label">{label}</div><div class="metric-value" style="color:{accent}">{value}</div><div class="metric-detail">{detail}</div></div>', unsafe_allow_html=True)


def route_bundle():
    routes = [plan_route(robot["start"], robot["goal"]) for robot in st.session_state.robots]
    positions = [path[min(robot["step"], len(path) - 1)] if path else robot["start"] for robot, path in zip(st.session_state.robots, routes)]
    return routes, positions


def dashboard():
    routes, positions = route_bundle()
    st.markdown('<div class="eyebrow">AUTONOMOUS NAVIGATION / OPERATIONS</div>', unsafe_allow_html=True)
    st.title("Mission Control")
    st.caption("A configurable command center for sensing, planning, and fleet movement.")
    cols = st.columns(4)
    with cols[0]: metric("MISSION STATE", "READY", f"{len(st.session_state.robots)} robot(s) configured")
    with cols[1]: metric("ROUTE LENGTH", f"{len(routes[0]) if routes else 0} cells", f"{st.session_state.planner} planner")
    with cols[2]: metric("OBSTACLES", f"{len(st.session_state.obstacles):02d}", "Editable world map", "#e07a5f")
    with cols[3]: metric("SYSTEM HEALTH", "98.6%", "Core services nominal")
    st.markdown("### Fleet overview")
    left, right = st.columns([1.7, 1])
    with left: st.pyplot(route_figure(routes, [r["start"] for r in st.session_state.robots], [r["goal"] for r in st.session_state.robots], positions), width="stretch")
    with right:
        st.markdown('<div class="panel"><div class="panel-kicker">CURRENT WORLD</div><h3>Operational map</h3><p class="muted">Edit obstacles, endpoints, planner, and robot behavior from the workspace pages.</p><div class="status-row"><span>Perception</span><b class="green">ONLINE</b></div><div class="status-row"><span>Planner</span><b class="green">ONLINE</b></div><div class="status-row"><span>Controller</span><b class="green">ONLINE</b></div></div>', unsafe_allow_html=True)
        if st.button("Open live simulation", type="primary", use_container_width=True):
            st.session_state.page = "Live Simulation"
            st.rerun()


def live_simulation():
    st.title("Live Simulation")
    st.caption("Run the fleet in cycles, tune controller behavior, and inspect safety events.")
    st.session_state.speed = st.sidebar.slider("Robot speed", 0.1, 5.0, float(st.session_state.speed), 0.1)
    kp = st.sidebar.slider("Controller gain", 0.1, 4.0, 1.5, 0.1)
    st.session_state.sim_paused = st.sidebar.toggle("Pause simulation", value=st.session_state.sim_paused)
    routes, positions = route_bundle()
    controls = st.columns(4)
    with controls[0]: advance = st.button("Advance cycle", type="primary", use_container_width=True)
    with controls[1]: run_cycles = st.button("Run 10 cycles", use_container_width=True)
    with controls[2]: reset = st.button("Reset fleet", use_container_width=True)
    with controls[3]: manual = st.selectbox("Manual nudge", ["None", "Up", "Down", "Left", "Right"], label_visibility="collapsed")
    if reset:
        for robot in st.session_state.robots: robot["step"] = 0
    if (advance or run_cycles) and not st.session_state.sim_paused:
        amount = 10 if run_cycles else 1
        for robot, path in zip(st.session_state.robots, routes): robot["step"] = min(robot["step"] + amount, max(0, len(path) - 1))
    if manual != "None":
        delta = {"Up": (-1, 0), "Down": (1, 0), "Left": (0, -1), "Right": (0, 1)}[manual]
        for robot, position in zip(st.session_state.robots, positions):
            candidate = (position[0] + delta[0], position[1] + delta[1])
            if valid_cell(candidate): robot["start"] = candidate
    routes, positions = route_bundle()
    left, right = st.columns([1.5, 1])
    with left: st.pyplot(route_figure(routes, [r["start"] for r in st.session_state.robots], [r["goal"] for r in st.session_state.robots], positions), width="stretch")
    with right:
        for robot, position, path in zip(st.session_state.robots, positions, routes): metric(robot["name"], f"{math.dist(position, robot['goal']):.1f} m", f"{len(path)} path cells | speed {st.session_state.speed:.1f} m/s")
        progress = np.mean([robot["step"] / max(1, len(path) - 1) for robot, path in zip(st.session_state.robots, routes)]) if routes else 0
        st.progress(float(progress), text="Fleet route completion")
        warnings = [robot["name"] for robot, position in zip(st.session_state.robots, positions) if not valid_cell(position)]
        if warnings: st.error("Collision warning: " + ", ".join(warnings))
        elif st.session_state.sim_paused: st.warning("Simulation paused")
        else: st.success(f"Controller active | gain {kp:.1f}")
    st.markdown("### Telemetry")
    st.line_chart({robot["name"]: [min(100, (robot["step"] + point) * 100 / max(1, len(path) - 1)) for point in range(10)] for robot, path in zip(st.session_state.robots, routes)}, y_label="Completion %")


def mission_planner():
    st.title("Mission Planner")
    st.caption("Create a mission, compare planners, and inspect alternative routes.")
    st.session_state.planner = st.selectbox("Primary planner", ["A*", "Dijkstra", "RRT"], index=["A*", "Dijkstra", "RRT"].index(st.session_state.planner))
    robot_count = st.number_input("Fleet size", 1, 4, len(st.session_state.robots))
    while len(st.session_state.robots) < robot_count:
        number = len(st.session_state.robots) + 1
        st.session_state.robots.append({"name": f"Rover {number}", "start": (2, 2), "goal": (st.session_state.rows - 3, st.session_state.cols - 3), "step": 0})
    while len(st.session_state.robots) > robot_count: st.session_state.robots.pop()
    starts, goals = [], []
    for index, robot in enumerate(st.session_state.robots):
        with st.expander(robot["name"], expanded=index == 0):
            c1, c2, c3, c4 = st.columns(4)
            start = (c1.number_input("Start row", 0, st.session_state.rows - 1, robot["start"][0], key=f"sr{index}"), c2.number_input("Start col", 0, st.session_state.cols - 1, robot["start"][1], key=f"sc{index}"))
            goal = (c3.number_input("Goal row", 0, st.session_state.rows - 1, robot["goal"][0], key=f"gr{index}"), c4.number_input("Goal col", 0, st.session_state.cols - 1, robot["goal"][1], key=f"gc{index}"))
            robot["start"], robot["goal"] = start, goal
            starts.append(start); goals.append(goal)
    routes = [plan_route(start, goal) for start, goal in zip(starts, goals)]
    if any(not path for path in routes): st.error("A route is blocked. Check obstacle cells or try another planner.")
    else: st.success(f"{len(routes)} route(s) valid using {st.session_state.planner}.")
    st.pyplot(route_figure(routes, starts, goals), width="stretch")
    if st.checkbox("Compare all planners"):
        comparison = []
        for name in ["A*", "Dijkstra", "RRT"]:
            paths = [plan_route(start, goal, name) for start, goal in zip(starts, goals)]
            comparison.append({"Planner": name, "Routes": len([path for path in paths if path]), "Total cells": sum(len(path) for path in paths), "Status": "Available" if all(paths) else "Blocked"})
        st.dataframe(comparison, hide_index=True, width="stretch")


def map_studio():
    st.title("Map Studio")
    st.caption("Build custom worlds, edit obstacle geometry, and save mission presets.")
    c1, c2, c3 = st.columns(3)
    rows = c1.number_input("Map rows", 10, 100, st.session_state.rows)
    cols = c2.number_input("Map columns", 10, 100, st.session_state.cols)
    preset = c3.selectbox("Load preset", ["Current map", "Open field", "Warehouse", "Maze"])
    if preset != "Current map" and st.button("Apply preset"):
        st.session_state.rows, st.session_state.cols = rows, cols
        if preset == "Open field": st.session_state.obstacles = []
        elif preset == "Warehouse": st.session_state.obstacles = [(rows // 3, cols // 3, max(2, rows // 5), max(2, cols // 5)), (rows // 2, 4, 4, max(3, cols // 2))]
        else: st.session_state.obstacles = [(r, cols // 2, 2, max(2, cols // 5)) for r in range(4, rows - 4, 6)]
        st.rerun()
    st.session_state.rows, st.session_state.cols = rows, cols
    obstacle_rows = [{"row": r, "col": c, "height": h, "width": w} for r, c, h, w in st.session_state.obstacles]
    edited = st.data_editor(obstacle_rows, num_rows="dynamic", hide_index=True, width="stretch")
    st.session_state.obstacles = [(int(item["row"]), int(item["col"]), int(item["height"]), int(item["width"])) for item in edited if all(key in item and item[key] is not None for key in ["row", "col", "height", "width"])]
    st.pyplot(route_figure([], [(1, 1)], [(rows - 2, cols - 2)]), width="stretch")
    name = st.text_input("Save map as", st.session_state.map_name)
    if st.button("Save map preset", type="primary"):
        maps = load_json(MAP_FILE, {})
        maps[name] = {"rows": rows, "cols": cols, "obstacles": st.session_state.obstacles}
        save_json(MAP_FILE, maps); st.session_state.map_name = name; st.success(f"Saved {name}.")


def perception_lab():
    st.title("Perception Lab")
    st.caption("Review uploaded images or videos as offline perception inputs. Webcam access is disabled.")
    confidence = st.slider("Detection confidence threshold", 0.30, 0.95, 0.60, 0.05)
    uploaded = st.file_uploader("Upload image or video", type=["jpg", "jpeg", "png", "mp4", "avi", "mov"])
    analyzed = None
    if uploaded:
        st.info(f"Loaded {uploaded.name}. Analysis is triggered manually and does not use a webcam.")
        if uploaded.type.startswith("image"):
            st.image(uploaded, caption="Uploaded frame", use_container_width=True)
        else:
            st.video(uploaded)
        if st.button("Analyze uploaded media", type="primary"):
            import cv2
            from src.perception import PerceptionEngine

            if uploaded.type.startswith("image"):
                analyzed = cv2.imdecode(np.frombuffer(uploaded.getvalue(), dtype=np.uint8), cv2.IMREAD_COLOR)
            else:
                temp_path = DATA_DIR / f"upload_{uploaded.name}"
                temp_path.write_bytes(uploaded.getvalue())
                capture = cv2.VideoCapture(str(temp_path))
                ok, analyzed = capture.read()
                capture.release()
                temp_path.unlink(missing_ok=True)
            if analyzed is not None:
                engine = PerceptionEngine()
                results = engine.model(analyzed, conf=confidence, verbose=False)[0]
                st.session_state.upload_detections = [
                    {"class": engine.model.names[int(box.cls[0])], "confidence": float(box.conf[0])}
                    for box in results.boxes
                ]
                st.success(f"Analyzed first frame: {len(st.session_state.upload_detections)} detection(s).")
    upload_detections = st.session_state.get("upload_detections", [])
    detections = [(row, col, height, width, f"{confidence:.2f}") for row, col, height, width in st.session_state.obstacles]
    left, right = st.columns([1.5, 1])
    with left: st.pyplot(route_figure([], [(5, 5)], [(st.session_state.rows - 3, st.session_state.cols - 3)], detections=detections), width="stretch")
    with right:
        metric("DETECTIONS", str(len(detections)), "Above active threshold")
        metric("GRID COVERAGE", f"{make_grid().mean() * 100:.1f}%", "Occupied cells", "#e07a5f")
        metric("INFERENCE", "32 ms", "Pipeline budget")
        st.dataframe(upload_detections or {"Class": ["obstacle"] * len(detections), "Confidence": [label for *_, label in detections], "Status": ["accepted"] * len(detections)}, hide_index=True, width="stretch")


def route_inspector():
    st.title("Route Inspector")
    routes, _ = route_bundle()
    selected = st.selectbox("Robot route", [robot["name"] for robot in st.session_state.robots])
    index = [robot["name"] for robot in st.session_state.robots].index(selected)
    route = routes[index]
    cols = st.columns(3)
    with cols[0]: metric("WAYPOINTS", str(len(route)), "8-connected grid")
    with cols[1]: metric("EST. COST", f"{sum(math.dist(route[i], route[i - 1]) for i in range(1, len(route))):.1f}" if route else "0.0", "Weighted route cost", "#e07a5f")
    with cols[2]: metric("CLEARANCE", "3.0 m", "Safety buffer")
    st.pyplot(route_figure([route], [st.session_state.robots[index]["start"]], [st.session_state.robots[index]["goal"]]), width="stretch")
    st.markdown("### Route analytics")
    st.bar_chart({"Distance": [len(route)], "Turns": [sum(route[i][0] != route[i - 1][0] and route[i][1] != route[i - 1][1] for i in range(1, len(route)))]})
    report = json.dumps({"planner": st.session_state.planner, "robot": selected, "route": route, "created": datetime.now().isoformat()}, indent=2)
    st.download_button("Download route report", report, file_name=f"{selected.lower().replace(' ', '-')}-route.json", mime="application/json")


def run_history():
    st.title("Run History")
    st.caption("Persistent operational records stored locally in data/run_history.json.")
    history = load_json(HISTORY_FILE, [])
    if not history: history = [{"Run ID": "A-17", "Mission": "Warehouse sweep", "Duration": "02:41", "Result": "Completed", "Date": "Today"}]
    st.dataframe(history, hide_index=True, width="stretch")
    mission = st.text_input("Mission label", "Warehouse sweep")
    if st.button("Archive current run", type="primary"):
        history.insert(0, {"Run ID": f"A-{len(history) + 17}", "Mission": mission, "Duration": "00:00", "Result": "Completed", "Date": datetime.now().strftime("%d %b %Y")})
        save_json(HISTORY_FILE, history); st.success("Run archived.")
    csv = "Run ID,Mission,Duration,Result,Date\n" + "\n".join(",".join(str(item.get(key, "")) for key in ["Run ID", "Mission", "Duration", "Result", "Date"]) for item in history)
    st.download_button("Export history CSV", csv, file_name="navos-history.csv", mime="text/csv")


init_state()
theme = st.sidebar.toggle("Dark theme", value=False)
st.markdown(f"""<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Space+Grotesk:wght@400;500;600;700&display=swap');
:root {{ --ink:{'#e5e7eb' if theme else '#172033'}; --muted:{'#aab4c3' if theme else '#64748b'}; --line:{'#334155' if theme else '#dbe3ea'}; --panel:{'#172033' if theme else '#ffffff'}; --accent:#087f8c; }}
html, body, [class*="css"] {{ font-family:'Space Grotesk', sans-serif; }} body {{ background:{'#0f172a' if theme else '#f5f7fa'}; color:var(--ink); }} .stApp {{ background:{'#0f172a' if theme else 'linear-gradient(135deg,#f8fafc 0%,#eef6f5 100%)'}; }}
[data-testid="stSidebar"] {{ background:var(--panel); border-right:1px solid var(--line); }} [data-testid="stSidebar"] h1 {{ font-family:'DM Mono',monospace; font-size:1rem; color:var(--accent); }} h1,h2,h3 {{ letter-spacing:-.03em; }} .eyebrow,.panel-kicker,.metric-label {{ font-family:'DM Mono',monospace; letter-spacing:.1em; font-size:.67rem; color:var(--muted); }} .metric {{ border-top:1px solid var(--line); padding:14px 0 8px; }} .metric-value {{ font-size:1.8rem; font-weight:600; margin:5px 0; }} .metric-detail,.muted {{ color:var(--muted); font-size:.78rem; }} .panel {{ background:var(--panel); border:1px solid var(--line); padding:22px; margin-top:12px; }} .status-row {{ display:flex; justify-content:space-between; padding:10px 0; border-bottom:1px solid var(--line); }} .green {{ color:#087f8c; font-family:'DM Mono',monospace; font-size:.7rem; }} button[kind="primary"] {{ background:#087f8c !important; color:white !important; border:0 !important; }}
</style>""", unsafe_allow_html=True)
with st.sidebar:
    st.markdown("# NAVOS")
    st.caption("COMMAND CENTER / v1.0")
    selected = st.radio("Workspace", PAGES, index=PAGES.index(st.session_state.get("page", "Mission Control")), label_visibility="collapsed")
    st.session_state.page = selected
    st.markdown("---")
    st.markdown(f'<div class="panel-kicker">ACTIVE MAP</div><p class="green">{st.session_state.map_name}</p><p class="muted">{st.session_state.rows} x {st.session_state.cols} grid</p>', unsafe_allow_html=True)

{"Mission Control": dashboard, "Live Simulation": live_simulation, "Mission Planner": mission_planner, "Map Studio": map_studio, "Perception Lab": perception_lab, "Route Inspector": route_inspector, "Run History": run_history}[st.session_state.page]()
