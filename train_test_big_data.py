import random
import numpy as np
import networkx as nx
import torch
import torch.optim as optim
import matplotlib.pyplot as plt

from bc.bc_config import RR_net_config, VR_net_config
from bc.graph_dataset.rr_graph2torch import rr_graph2torch
from bc.graph_dataset.vr_graph2torch import vr_graph2torch

from env.env_config import MAXIMUM_SEATS
from env.models.ride import Ride
from env.models.request import Request
from env.models.vehicle import Vehicle
from env.services.rr_match_observe import request_request_observe
from env.services.vr_match_observe import vehicle_ride_observe

from models.Bipartite_RGGCN import Bipartite_RGGCNModel
from models.RGGCN import ResidualGatedGCNModel


class DynamicSimulator:
    """Simulator for vehicles and ride requests."""
    def __init__(self, num_vehicles=100, num_requests=500, max_time=300):
        self.num_vehicles = num_vehicles
        self.num_requests = num_requests
        self.max_time = max_time
        self.time = 0

        # Initialize vehicles with random positions
        self.vehicles = {
            i: Vehicle(i, 0, [random.uniform(0, 200), random.uniform(0, 200)])
            for i in range(num_vehicles)
        }

        self.requests = {}
        self.active_requests = set()
        self.next_request_id = 0
        self.requests_matched = {}
        self.next_ride_id = 0

    def step(self):
        """Advance simulator by one time step."""
        self.time += 1
        for v in self.vehicles.values():
            v.step(self.time)
        self._remove_expired_requests()
        self._generate_new_requests()

    def _generate_new_requests(self):
        """Generate new ride requests randomly."""
        if self.next_request_id >= self.num_requests:
            return
        remaining = self.num_requests - self.next_request_id
        new_num = random.randint(1, min(50, remaining))
        for _ in range(new_num):
            rid = self.next_request_id
            start_loc = [random.uniform(0, 200), random.uniform(0, 200)]
            end_loc = [random.uniform(0, 200), random.uniform(0, 200)]
            n_passenger = random.randint(1, MAXIMUM_SEATS - 1)
            req = Request(rid, self.time, start_loc, end_loc, n_passenger)
            self.requests[rid] = req
            self.active_requests.add(rid)
            self.requests_matched[rid] = False
            self.next_request_id += 1

    def _remove_expired_requests(self):
        """Remove requests that are no longer valid."""
        expired = [rid for rid in self.active_requests if not self.requests[rid].whether_exist()]
        for rid in expired:
            self.active_requests.remove(rid)
            self.requests[rid].expired = True

    def get_current_requests(self):
        return [self.requests[rid] for rid in self.active_requests]

    def get_idle_vehicles(self):
        return [v for v in self.vehicles.values() if v.state == 'idle']

    def all_done(self):
        return self.time >= self.max_time or (
            len(self.active_requests) == 0 and self.next_request_id >= self.num_requests
        )


def generate_rr_vr_graphs(sim: DynamicSimulator):
    """Generate RR and VR graphs and compute MWM for both."""
    requests = sim.get_current_requests()
    vehicles = sim.get_idle_vehicles()
    if not requests or not vehicles:
        return None

    # --- RR Graph ---
    rr_graph = request_request_observe(requests)
    rr_MWM = nx.max_weight_matching(rr_graph, maxcardinality=True)

    rides = []
    matched_nodes = set()
    rr_nodes = list(rr_graph.nodes)
    node_to_request = {node: requests[idx] for idx, node in enumerate(rr_nodes)}

    # RR Matching: pair requests
    for u, v in rr_MWM:
        if u not in matched_nodes and v not in matched_nodes:
            ride_requests = [node_to_request[u], node_to_request[v]]
            ride = Ride(ride_requests, sim.next_ride_id)
            ride.assigned = False
            rides.append(ride)
            sim.next_ride_id += 1
            matched_nodes.update([u, v])

    # Unmatched single requests
    for node in rr_nodes:
        if node not in matched_nodes:
            ride = Ride([node_to_request[node]], sim.next_ride_id)
            ride.assigned = False
            rides.append(ride)
            sim.next_ride_id += 1

    # --- VR Graph ---
    # Vehicle nodes: 0..num_vehicles-1
    # Ride nodes: num_vehicles..num_vehicles+len(rides)-1
    vr_graph = vehicle_ride_observe(vehicles, rides)
    vr_MWM = nx.max_weight_matching(vr_graph, maxcardinality=True)

    return rr_graph, rr_MWM, vr_graph, vr_MWM, requests, rides, vehicles


def update_requests_matched(sim: DynamicSimulator, rr_MWM, requests):
    """Update which requests are matched in RR stage."""
    matched_indices = set()
    for u, v in rr_MWM:
        matched_indices.add(u)
        matched_indices.add(v)
    for idx in matched_indices:
        sim.requests_matched[requests[idx].id] = True


def assign_rides_to_vehicles(sim: DynamicSimulator, vr_MWM, vehicles, rides):
    """Assign rides to vehicles according to VR matching."""
    num_vehicles = len(vehicles)
    for u, v in vr_MWM:
        if u < num_vehicles and v >= num_vehicles:
            vehicle = vehicles[u]
            ride = rides[v - num_vehicles]
        elif v < num_vehicles and u >= num_vehicles:
            vehicle = vehicles[v]
            ride = rides[u - num_vehicles]
        else:
            continue

        if getattr(ride, "assigned", False):
            continue
        if vehicle.state == 'idle':
            vehicle.set_occupied(ride)
            ride.assigned = True
            for req in ride.requests:
                sim.requests_matched[req.id] = True


def count_unmatched_timeout_requests(sim: DynamicSimulator):
    """
    Count the number of requests that remain unmatched
    and have already expired at the end of the simulation.
    """
    return sum(
        1 for rid, req in sim.requests.items()
        if not sim.requests_matched.get(rid, False) and not req.whether_exist()
    )


# -------------------- NEW: helper to collect MWM-matched requests --------------------
def collect_mwm_matched_request_ids(vr_MWM, vehicles, rides):
    """
    Given a VR MWM on the bipartite graph (vehicles vs. rides),
    return the set of request IDs that would be served by the MWM baseline.
    """
    matched_req_ids = set()
    num_vehicles = len(vehicles)
    for u, v in vr_MWM:
        # Identify which endpoint is the ride node, then map to Ride object.
        if u < num_vehicles and v >= num_vehicles:
            ride = rides[v - num_vehicles]
        elif v < num_vehicles and u >= num_vehicles:
            ride = rides[u - num_vehicles]
        else:
            continue
        for req in ride.requests:
            matched_req_ids.add(req.id)
    return matched_req_ids
# ------------------------------------------------------------------------------------


def greedy_matching(prob_matrix, rides):
    """Greedy VR matching from predicted probabilities."""
    prob_np = prob_matrix.detach().cpu().numpy()
    num_vehicles, num_rides = prob_np.shape
    matched_pairs = []
    used_left = set()
    used_right = set()
    edges = [(i, j, prob_np[i, j]) for i in range(num_vehicles) for j in range(num_rides) if not getattr(rides[j], 'assigned', False)]
    edges.sort(key=lambda x: x[2], reverse=True)
    for i, j, _ in edges:
        if i not in used_left and j not in used_right:
            matched_pairs.append((i, j + num_vehicles))  # Ride node ID offset by num_vehicles
            used_left.add(i)
            used_right.add(j)
    return matched_pairs


def compare_matchings(mwm_pairs, model_pairs, print_diff=True):
    """Check if two matchings are the same."""
    mwm_set = {tuple(sorted(p)) for p in mwm_pairs}
    model_set = {tuple(sorted(p)) for p in model_pairs}
    if mwm_set == model_set:
        return True
    else:
        if print_diff:
            print("Matchings differ.")
            print("In MWM but not in model:", mwm_set - model_set)
            print("In model but not in MWM:", model_set - mwm_set)
        return False


def train_one_epoch(rr_model, vr_model, optimizer_rr, optimizer_vr, sim: DynamicSimulator):
    """Run one training epoch on the given simulator."""
    rr_model.train()
    vr_model.train()
    rr_loss_total, vr_loss_total = 0, 0
    steps = 0
    match_agree_count = 0

    # NEW: collect request IDs that MWM would serve across the whole epoch
    mwm_matched_request_ids = set()

    while not sim.all_done():
        sim.step()
        out = generate_rr_vr_graphs(sim)
        if out is None:
            continue
        rr_graph, rr_MWM, vr_graph, vr_MWM, requests, rides, vehicles = out

        rr_data = rr_graph2torch(rr_graph, MWM=rr_MWM)
        vr_data = vr_graph2torch(vr_graph, MWM=vr_MWM)

        optimizer_rr.zero_grad()
        optimizer_vr.zero_grad()
        _, rr_loss = rr_model(rr_data)
        _, vr_loss = vr_model(vr_data)
        rr_loss.backward()
        vr_loss.backward()
        optimizer_rr.step()
        optimizer_vr.step()

        # VR prediction step
        pred_vr_matrix = vr_model.predict(vr_data)
        predicted_vr_MWM = greedy_matching(pred_vr_matrix, rides)

        if compare_matchings(list(vr_MWM), predicted_vr_MWM, print_diff=False):
            match_agree_count += 1

        # Update simulator states (this is the MODEL outcome)
        update_requests_matched(sim, rr_MWM, requests)
        assign_rides_to_vehicles(sim, predicted_vr_MWM, vehicles, rides)

        # NEW: accumulate which requests MWM would have served (baseline outcome)
        mwm_matched_request_ids.update(collect_mwm_matched_request_ids(vr_MWM, vehicles, rides))

        rr_loss_total += rr_loss.item()
        vr_loss_total += vr_loss.item()
        steps += 1

    # --- Compute final unmatched rate after the whole simulation (MODEL) ---
    unmatched_count = count_unmatched_timeout_requests(sim)
    total_requests = sim.next_request_id
    unmatched_rate = unmatched_count / total_requests if total_requests > 0 else 0
    agreement_rate = match_agree_count / steps if steps > 0 else 0

    # --- NEW: Compute final unmatched rate for MWM baseline (DO NOT modify sim states) ---
    mwm_unmatched_count = sum(
        1 for rid, req in sim.requests.items()
        if (rid not in mwm_matched_request_ids) and (not req.whether_exist())
    )
    mwm_unmatched_rate = mwm_unmatched_count / total_requests if total_requests > 0 else 0

    return rr_loss_total / steps, vr_loss_total / steps, unmatched_rate, agreement_rate, mwm_unmatched_rate


@torch.no_grad()
def evaluate_one_epoch(rr_model, vr_model, sim: DynamicSimulator):
    """Run one evaluation epoch on the given simulator."""
    rr_model.eval()
    vr_model.eval()
    rr_loss_total, vr_loss_total = 0, 0
    steps = 0
    match_agree_count = 0

    # NEW: collect request IDs that MWM would serve across the whole epoch
    mwm_matched_request_ids = set()

    while not sim.all_done():
        sim.step()
        out = generate_rr_vr_graphs(sim)
        if out is None:
            continue
        rr_graph, rr_MWM, vr_graph, vr_MWM, requests, rides, vehicles = out

        rr_data = rr_graph2torch(rr_graph, MWM=rr_MWM)
        vr_data = vr_graph2torch(vr_graph, MWM=vr_MWM)
        _, rr_loss = rr_model(rr_data)
        _, vr_loss = vr_model(vr_data)

        # VR prediction step
        pred_vr_matrix = vr_model.predict(vr_data)
        predicted_vr_MWM = greedy_matching(pred_vr_matrix, rides)

        if compare_matchings(list(vr_MWM), predicted_vr_MWM, print_diff=False):
            match_agree_count += 1

        # Update simulator states (MODEL outcome)
        update_requests_matched(sim, rr_MWM, requests)
        assign_rides_to_vehicles(sim, predicted_vr_MWM, vehicles, rides)

        # NEW: accumulate MWM-served requests (baseline)
        mwm_matched_request_ids.update(collect_mwm_matched_request_ids(vr_MWM, vehicles, rides))

        rr_loss_total += rr_loss.item()
        vr_loss_total += vr_loss.item()
        steps += 1

    # --- Compute final unmatched rate after the whole simulation (MODEL) ---
    unmatched_count = count_unmatched_timeout_requests(sim)
    total_requests = sim.next_request_id
    unmatched_rate = unmatched_count / total_requests if total_requests > 0 else 0
    agreement_rate = match_agree_count / steps if steps > 0 else 0

    # --- NEW: Compute final unmatched rate for MWM baseline ---
    mwm_unmatched_count = sum(
        1 for rid, req in sim.requests.items()
        if (rid not in mwm_matched_request_ids) and (not req.whether_exist())
    )
    mwm_unmatched_rate = mwm_unmatched_count / total_requests if total_requests > 0 else 0

    return rr_loss_total / steps, vr_loss_total / steps, unmatched_rate, agreement_rate, mwm_unmatched_rate


def plot_dual_axis_loss(rr_losses, vr_losses, title):
    """Plot RR and VR losses on dual axis."""
    fig, ax1 = plt.subplots(figsize=(6, 4))
    ax2 = ax1.twinx()
    ax1.plot(rr_losses, 'g-', label='RR Loss')
    ax2.plot(vr_losses, 'b-', label='VR Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('RR Loss', color='g')
    ax2.set_ylabel('VR Loss', color='b')
    ax1.set_title(title)
    ax1.grid(True)
    fig.tight_layout()
    plt.show()


def main():
    rr_model = ResidualGatedGCNModel(RR_net_config)
    vr_model = Bipartite_RGGCNModel(VR_net_config)
    optimizer_rr = optim.Adam(rr_model.parameters(), lr=1e-3)
    optimizer_vr = optim.Adam(vr_model.parameters(), lr=1e-3)

    rr_losses, vr_losses = [], []
    val_rr_losses, val_vr_losses = [], []
    test_rr_losses, test_vr_losses = [], []

    train_unmatched_rates, val_unmatched_rates, test_unmatched_rates = [], [], []
    train_agreement_rates, val_agreement_rates, test_agreement_rates = [], [], []

    # NEW: store MWM baseline unmatched rates
    train_mwm_unmatched_rates, val_mwm_unmatched_rates, test_mwm_unmatched_rates = [], [], []

    epochs = 300
    for epoch in range(epochs):
        train_sim = DynamicSimulator(num_vehicles=120, num_requests=500, max_time=300)
        val_sim = DynamicSimulator(num_vehicles=120, num_requests=500, max_time=300)
        test_sim = DynamicSimulator(num_vehicles=120, num_requests=500, max_time=300)

        rr_loss, vr_loss, train_unmatched, train_agree, train_mwm_unmatched = train_one_epoch(
            rr_model, vr_model, optimizer_rr, optimizer_vr, train_sim
        )
        val_rr, val_vr, val_unmatched, val_agree, val_mwm_unmatched = evaluate_one_epoch(rr_model, vr_model, val_sim)
        test_rr, test_vr, test_unmatched, test_agree, test_mwm_unmatched = evaluate_one_epoch(rr_model, vr_model, test_sim)

        rr_losses.append(rr_loss)
        vr_losses.append(vr_loss)
        val_rr_losses.append(val_rr)
        val_vr_losses.append(val_vr)
        test_rr_losses.append(test_rr)
        test_vr_losses.append(test_vr)

        train_unmatched_rates.append(train_unmatched)
        val_unmatched_rates.append(val_unmatched)
        test_unmatched_rates.append(test_unmatched)

        train_agreement_rates.append(train_agree)
        val_agreement_rates.append(val_agree)
        test_agreement_rates.append(test_agree)

        # NEW: push MWM unmatched rates
        train_mwm_unmatched_rates.append(train_mwm_unmatched)
        val_mwm_unmatched_rates.append(val_mwm_unmatched)
        test_mwm_unmatched_rates.append(test_mwm_unmatched)

        if (epoch + 1) % 60 == 0:
            print(f"\n=== Epoch {epoch+1} ===")
            print(f"Train Loss: RR={rr_loss:.4f}, VR={vr_loss:.4f}, "
                  f"Unmatched={train_unmatched:.4f}, MWM_Unmatched={train_mwm_unmatched:.4f}, Agreement={train_agree:.4f}")
            print(f"Val   Loss: RR={val_rr:.4f}, VR={val_vr:.4f}, "
                  f"Unmatched={val_unmatched:.4f}, MWM_Unmatched={val_mwm_unmatched:.4f}, Agreement={val_agree:.4f}")
            print(f"Test  Loss: RR={test_rr:.4f}, VR={test_vr:.4f}, "
                  f"Unmatched={test_unmatched:.4f}, MWM_Unmatched={test_mwm_unmatched:.4f}, Agreement={test_agree:.4f}")

    # --- Loss curves ---
    plot_dual_axis_loss(rr_losses, vr_losses, title="Train RR & VR Loss Over Epochs")
    plot_dual_axis_loss(val_rr_losses, val_vr_losses, title="Validation RR & VR Loss Over Epochs")
    plot_dual_axis_loss(test_rr_losses, test_vr_losses, title="Test RR & VR Loss Over Epochs")

    # --- Agreement curves ---
    plt.figure()
    plt.plot(train_agreement_rates, label="Train Agreement Rate")
    plt.plot(val_agreement_rates, label="Validation Agreement Rate")
    plt.plot(test_agreement_rates, label="Test Agreement Rate")
    plt.xlabel("Epoch")
    plt.ylabel("Agreement Rate")
    plt.title("VR Matching Agreement Rate")
    plt.legend()
    plt.grid(True)
    plt.show()

    # --- Unmatched rate curves (MODEL) ---
    plt.figure()
    plt.plot(train_unmatched_rates, label="Train Unmatched Rate")
    plt.plot(val_unmatched_rates, label="Validation Unmatched Rate")
    plt.plot(test_unmatched_rates, label="Test Unmatched Rate")
    plt.xlabel("Epoch")
    plt.ylabel("Unmatched Rate")
    plt.title("Unmatched Rate Over Epochs (Model)")
    plt.ylim(0.5, 1.0) 
    plt.legend()
    plt.grid(True)
    plt.show()

    # --- NEW: MWM unmatched rate curves (separate figure as requested) ---
    plt.figure()
    plt.plot(train_mwm_unmatched_rates, label="Train MWM Unmatched Rate")
    plt.plot(val_mwm_unmatched_rates, label="Validation MWM Unmatched Rate")
    plt.plot(test_mwm_unmatched_rates, label="Test MWM Unmatched Rate")
    plt.xlabel("Epoch")
    plt.ylabel("MWM Unmatched Rate")
    plt.title("Unmatched Rate Over Epochs (MWM Baseline)")
    plt.ylim(0.5, 1.0) 
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()
