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


# -------- Dynamic simulator class --------
class DynamicSimulator:
    def __init__(self, num_vehicles=100, num_requests=500, max_time=300):
        self.num_vehicles = num_vehicles
        self.num_requests = num_requests
        self.max_time = max_time
        self.time = 0

        # Initialize vehicles with random locations and idle state
        self.vehicles = {
            i: Vehicle(i, 0, [random.uniform(0, 200), random.uniform(0, 200)])
            for i in range(num_vehicles)
        }

        self.requests = {}
        self.active_requests = set()
        self.next_request_id = 0
        # Track whether each request is matched
        self.requests_matched = {}

    def step(self):
        """
        Advance simulation time by one step.
        Update vehicles (move, update states),
        remove expired requests,
        generate new requests.
        """
        self.time += 1
        for v in self.vehicles.values():
            v.step(self.time)
        self._remove_expired_requests()
        self._generate_new_requests()

    def _generate_new_requests(self):
        """
        Generate a random number (1 to 50) of new requests at current time step
        until reaching the total num_requests limit.
        """
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
        """
        Remove requests from active set if they have expired (no longer valid).
        """
        expired = [rid for rid in self.active_requests if not self.requests[rid].whether_exist()]
        for rid in expired:
            self.active_requests.remove(rid)
            self.requests[rid].expired = True

    def get_current_requests(self):
        """
        Return a list of currently active (unmatched and valid) requests.
        """
        return [self.requests[rid] for rid in self.active_requests]

    def get_idle_vehicles(self):
        """
        Return a list of vehicles currently idle (available to accept new rides).
        """
        return [v for v in self.vehicles.values() if v.state == 'idle']

    def all_done(self):
        """
        Return True if simulation time exceeded max_time or
        no active requests remain and all requests have been generated.
        """
        return self.time >= self.max_time or (len(self.active_requests) == 0 and self.next_request_id >= self.num_requests)


# -------- Generate graphs and matchings --------
def generate_rr_vr_graphs(sim: DynamicSimulator):
    """
    Generate request-request (RR) and vehicle-ride (VR) graphs for matching.
    Returns:
        rr_graph, rr_MWM, vr_graph, vr_MWM, requests, rides, vehicles
    """
    requests = sim.get_current_requests()
    vehicles = sim.get_idle_vehicles()
    if not requests or not vehicles:
        return None

    rr_graph = request_request_observe(requests)
    rr_MWM = nx.max_weight_matching(rr_graph, maxcardinality=True)

    # Create rides from RR matched pairs and single unmatched requests
    rides = []
    matched = set()
    ride_id = 0
    for i, j in rr_MWM:
        if i not in matched and j not in matched:
            rides.append(Ride([requests[i], requests[j]], ride_id))
            matched.update([i, j])
            ride_id += 1
    for i, req in enumerate(requests):
        if i not in matched:
            rides.append(Ride([req], ride_id))
            ride_id += 1

    vr_graph = vehicle_ride_observe(vehicles, rides)
    vr_MWM = nx.max_weight_matching(vr_graph, maxcardinality=True)

    return rr_graph, rr_MWM, vr_graph, vr_MWM, requests, rides, vehicles


def update_requests_matched(sim: DynamicSimulator, rr_MWM, requests):
    """
    Mark matched requests as True in sim.requests_matched
    """
    matched_indices = set()
    for u, v in rr_MWM:
        matched_indices.add(u)
        matched_indices.add(v)
    for idx in matched_indices:
        sim.requests_matched[requests[idx].id] = True


def assign_rides_to_vehicles(sim: DynamicSimulator, vr_MWM, vehicles, rides):
    """
    Assign matched rides to vehicles and mark their requests as matched.
    """
    vehicle_id_map = {v.id: v for v in vehicles}
    ride_id_map = {r.id: r for r in rides}

    for u, v in vr_MWM:
        if u in vehicle_id_map and v in ride_id_map:
            vehicle = vehicle_id_map[u]
            ride = ride_id_map[v]
        elif v in vehicle_id_map and u in ride_id_map:
            vehicle = vehicle_id_map[v]
            ride = ride_id_map[u]
        else:
            continue

        vehicle.set_occupied(ride)

        for req in ride.requests:
            sim.requests_matched[req.id] = True


def count_unmatched_timeout_requests(sim: DynamicSimulator):
    """
    Count how many requests have timed out (expired and unmatched).
    """
    return sum(1 for rid, req in sim.requests.items() if not sim.requests_matched.get(rid, False) and not req.whether_exist())



# -------- Greedy matching function --------
def greedy_matching(prob_matrix):
    """
    Args:
        prob_matrix: torch.Tensor of shape (num_left_nodes, num_right_nodes)
                     Represents predicted edge probabilities.
    Returns:
        List of matched pairs [(left_idx, right_idx), ...] chosen greedily by highest probability
    Method:
        Sort all edges by probability descending, pick edges one-by-one if no conflicts.
    """
    prob_np = prob_matrix.detach().cpu().numpy()
    num_left, num_right = prob_np.shape
    matched_pairs = []
    used_left = set()
    used_right = set()

    edges = [(i, j, prob_np[i, j]) for i in range(num_left) for j in range(num_right)]
    edges.sort(key=lambda x: x[2], reverse=True)

    for i, j, p in edges:
        if i not in used_left and j not in used_right:
            matched_pairs.append((i, j))
            used_left.add(i)
            used_right.add(j)
    return matched_pairs



# -------- Training loop --------
def train_one_epoch(rr_model, vr_model, optimizer_rr, optimizer_vr, sim: DynamicSimulator):
    rr_model.train()
    vr_model.train()

    rr_loss_total, vr_loss_total = 0, 0
    steps = 0

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

        # Forward pass: compute losses
        _, rr_loss = rr_model(rr_data)
        _, vr_loss = vr_model(vr_data)

        # Backprop and optimize
        rr_loss.backward()
        vr_loss.backward()
        optimizer_rr.step()
        optimizer_vr.step()

        # Use model prediction to obtain VR matching, instead of Hungarian
        pred_vr_matrix = vr_model.predict(vr_data)  # shape: (num_vehicles, num_rides)
        predicted_vr_MWM = greedy_matching(pred_vr_matrix)

        # RR matching still uses MWM (could also use model prediction here if needed)
        update_requests_matched(sim, rr_MWM, requests)
        assign_rides_to_vehicles(sim, predicted_vr_MWM, vehicles, rides)

        rr_loss_total += rr_loss.item()
        vr_loss_total += vr_loss.item()
        steps += 1

    unmatched_count = count_unmatched_timeout_requests(sim)
    total_requests = sim.next_request_id
    unmatched_rate = unmatched_count / total_requests if total_requests > 0 else 0

    avg_rr_loss = rr_loss_total / steps if steps > 0 else 0
    avg_vr_loss = vr_loss_total / steps if steps > 0 else 0

    return avg_rr_loss, avg_vr_loss, unmatched_rate


# -------- Evaluation loop --------
@torch.no_grad()
def evaluate_one_epoch(rr_model, vr_model, sim: DynamicSimulator):
    rr_model.eval()
    vr_model.eval()

    rr_loss_total, vr_loss_total = 0, 0
    steps = 0

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

        pred_vr_matrix = vr_model.predict(vr_data)
        predicted_vr_MWM = greedy_matching(pred_vr_matrix)

        update_requests_matched(sim, rr_MWM, requests)
        assign_rides_to_vehicles(sim, predicted_vr_MWM, vehicles, rides)

        rr_loss_total += rr_loss.item()
        vr_loss_total += vr_loss.item()
        steps += 1

    unmatched_count = count_unmatched_timeout_requests(sim)
    total_requests = sim.next_request_id
    unmatched_rate = unmatched_count / total_requests if total_requests > 0 else 0

    avg_rr_loss = rr_loss_total / steps if steps > 0 else 0
    avg_vr_loss = vr_loss_total / steps if steps > 0 else 0

    return avg_rr_loss, avg_vr_loss, unmatched_rate


# -------- Plotting function --------
def plot_curves(data_dict, ylabel, title, ylim = None):
    plt.figure(figsize=(6, 4))
    for label, values in data_dict.items():
        plt.plot(values, label=label)
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(title)
    if ylim:
        plt.ylim(*ylim)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# -------- Main entry point --------
def main():
    rr_model = ResidualGatedGCNModel(RR_net_config)
    vr_model = Bipartite_RGGCNModel(VR_net_config)
    optimizer_rr = optim.Adam(rr_model.parameters(), lr=1e-3)
    optimizer_vr = optim.Adam(vr_model.parameters(), lr=1e-3)

    rr_losses, vr_losses = [], []
    val_rr_losses, val_vr_losses = [], []
    test_rr_losses, test_vr_losses = [], []
    train_unmatched_rates, val_unmatched_rates, test_unmatched_rates = [], [], []

    epochs = 150

    for epoch in range(epochs):
        train_sim = DynamicSimulator(num_vehicles=120, num_requests=500, max_time=300)
        val_sim = DynamicSimulator(num_vehicles=30, num_requests=100, max_time=100)
        test_sim = DynamicSimulator(num_vehicles=30, num_requests=100, max_time=100)

        # Training / Validation / Test
        rr_loss, vr_loss, train_unmatched = train_one_epoch(rr_model, vr_model, optimizer_rr, optimizer_vr, train_sim)
        val_rr, val_vr, val_unmatched = evaluate_one_epoch(rr_model, vr_model, val_sim)
        test_rr, test_vr, test_unmatched = evaluate_one_epoch(rr_model, vr_model, test_sim)

        # Append results
        rr_losses.append(rr_loss)
        vr_losses.append(vr_loss)
        val_rr_losses.append(val_rr)
        val_vr_losses.append(val_vr)
        test_rr_losses.append(test_rr)
        test_vr_losses.append(test_vr)
        train_unmatched_rates.append(train_unmatched)
        val_unmatched_rates.append(val_unmatched)
        test_unmatched_rates.append(test_unmatched)

        if (epoch + 1) % 30 == 0:
            print(f"[Epoch {epoch+1}] Train Loss: RR={rr_loss:.4f}, VR={vr_loss:.4f}, Unmatched={train_unmatched:.4f}")
            print(f"              Val Loss:   RR={val_rr:.4f}, VR={val_vr:.4f}, Unmatched={val_unmatched:.4f}")
            print(f"              Test Loss:  RR={test_rr:.4f}, VR={test_vr:.4f}, Unmatched={test_unmatched:.4f}")

    # plot
    plot_curves({"Train RR": rr_losses, "Train VR": vr_losses}, ylabel="Loss", title="Train Loss Over Epochs")
    plot_curves({"Val RR": val_rr_losses, "Val VR": val_vr_losses}, ylabel="Loss", title="Validation Loss Over Epochs")
    plot_curves({"Test RR": test_rr_losses, "Test VR": test_vr_losses}, ylabel="Loss", title="Test Loss Over Epochs")

    plot_curves({
        "Train": train_unmatched_rates,
        "Val": val_unmatched_rates,
        "Test": test_unmatched_rates
    }, ylabel="Unmatched Rate", title="Unmatched Rate Over Epochs", ylim=(0, 1))



if __name__ == "__main__":
    main()
