import sys
import time
import gurobipy as gp
from gurobipy import GRB
import argparse
from collections import defaultdict, Counter


def parse_input(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    n = int(lines[0].strip())
    photos = []
    
    for i in range(1, n+1):
        parts = lines[i].strip().split()
        orientation = parts[0]  # 'H' or 'V'
        num_tags = int(parts[1])
        tags = parts[2:2+num_tags]
        photos.append({
            'id': i-1,  # 0-indexed
            'orientation': orientation,
            'tags': set(tags)
        })
    
    return photos


def compute_interest_factor(tags1, tags2):
    common = len(tags1 & tags2)
    only_in_1 = len(tags1 - tags2)
    only_in_2 = len(tags2 - tags1)
    return min(common, only_in_1, only_in_2)


def create_slides_from_photos(photos):

    h_photos = [p for p in photos if p['orientation'] == 'H']
    v_photos = [p for p in photos if p['orientation'] == 'V']
    
    # Create slides for horizontal photos
    h_slides = [{
        'photos': [p['id']],
        'tags': p['tags']
    } for p in h_photos]
    
    # For vertical photos, we'll handle them in the optimization model
    # We just return them separately
    
    return h_slides, v_photos


def optimize_slideshow(h_slides, v_photos):
    # Phase 1: Create vertical photo pairs using a greedy approach
    # For larger instances, this is more efficient than trying to optimize pairing and sequencing together
    print(f"Creating pairs for {len(v_photos)} vertical photos...")
    v_pairs = []
    used_v_photos = set()
    
    # Sort vertical photos by number of tags (descending) to start with more informative photos
    v_photos_sorted = sorted(v_photos, key=lambda p: len(p['tags']), reverse=True)
    
    # Greedy pairing strategy
    for i, photo_i in enumerate(v_photos_sorted):
        if photo_i['id'] in used_v_photos:
            continue
            
        # Find best matching photo (not used yet)
        best_match = None
        best_score = -1
        
        for j, photo_j in enumerate(v_photos_sorted[i+1:], i+1):
            if photo_j['id'] in used_v_photos:
                continue
                
            # Calculate a diversity score that rewards pairs with diverse tags
            # (which can lead to higher interest factors)
            combined_tags = photo_i['tags'] | photo_j['tags']
            score = len(combined_tags)
            
            if score > best_score:
                best_score = score
                best_match = photo_j
        
        # If found a match, create a pair
        if best_match:
            v_pairs.append({
                'photos': [photo_i['id'], best_match['id']],
                'tags': photo_i['tags'] | best_match['tags']
            })
            used_v_photos.add(photo_i['id'])
            used_v_photos.add(best_match['id'])
    
    print(f"Created {len(v_pairs)} vertical photo pairs")
    
    # Combine horizontal slides and vertical pairs
    all_slides = h_slides + v_pairs
    n_slides = len(all_slides)
    
    if n_slides == 0:
        print("No slides created. Returning empty solution.")
        return []
    
    # Phase 2: Optimize the sequence of slides
    print(f"Optimizing sequence for {n_slides} slides...")
    model = gp.Model("Photo_Slideshow")
    
    # Start timing
    start_time = time.time()
    
    # Create variables for slide positions
    pos_vars = {}
    for i in range(n_slides):
        for pos in range(n_slides):
            pos_vars[(i, pos)] = model.addVar(vtype=GRB.BINARY, name=f"slide_{i}_pos_{pos}")
    
    # For larger models, compute interest factors on demand
    # instead of precomputing all of them
    interest_cache = {}
    def get_interest_factor(i, j):
        if (i, j) not in interest_cache:
            tags_i = all_slides[i]['tags']
            tags_j = all_slides[j]['tags']
            interest_cache[(i, j)] = compute_interest_factor(tags_i, tags_j)
        return interest_cache[(i, j)]
    
    # Constraint: Each slide must be placed in exactly one position
    for i in range(n_slides):
        model.addConstr(gp.quicksum(pos_vars[(i, pos)] for pos in range(n_slides)) == 1, f"slide_{i}_placed")
    
    # Constraint: Each position must have exactly one slide
    for pos in range(n_slides):
        model.addConstr(gp.quicksum(pos_vars[(i, pos)] for i in range(n_slides)) == 1, f"pos_{pos}_filled")
    
    # Create transition variables to linearize the objective
    transition_vars = {}
    for pos in range(n_slides - 1):
        for i in range(n_slides):
            for j in range(n_slides):
                if i != j:
                    # Create variable representing that slide i is at position pos and slide j is at position pos+1
                    transition_vars[(i, j, pos)] = model.addVar(vtype=GRB.BINARY, name=f"trans_{i}_{j}_{pos}")
                    # Enforce: transition_vars[(i, j, pos)] = pos_vars[(i, pos)] * pos_vars[(j, pos+1)]
                    model.addConstr(transition_vars[(i, j, pos)] <= pos_vars[(i, pos)], f"trans_{i}_{j}_{pos}_1")
                    model.addConstr(transition_vars[(i, j, pos)] <= pos_vars[(j, pos+1)], f"trans_{i}_{j}_{pos}_2")
                    model.addConstr(transition_vars[(i, j, pos)] >= pos_vars[(i, pos)] + pos_vars[(j, pos+1)] - 1, f"trans_{i}_{j}_{pos}_3")
    
    # Objective: Maximize the sum of interest factors between consecutive slides
    objective_terms = []
    for pos in range(n_slides - 1):
        for i in range(n_slides):
            for j in range(n_slides):
                if i != j:
                    factor = get_interest_factor(i, j)
                    objective_terms.append(transition_vars[(i, j, pos)] * factor)
    
    # Set the objective to maximize the sum of interest factors
    model.setObjective(gp.quicksum(objective_terms), GRB.MAXIMIZE)
    
    # Set Gurobi parameters for performance
    model.setParam('TimeLimit', 300)  # 5 minute time limit
    model.setParam('MIPFocus', 1)     # Focus on finding feasible solutions
    model.setParam('Threads', 0)      # Use all available threads
    model.setParam('OutputFlag', 1)   # Show progress
    
    # For large models, use heuristics to find good solutions quickly
    if n_slides > 100:
        model.setParam('Heuristics', 0.8)
        model.setParam('MIPGap', 0.05)  # Accept solutions within 5% of optimal
    
    # Optimize the model
    model.optimize()
    
    end_time = time.time()
    print(f"Optimization completed in {end_time - start_time:.2f} seconds")
    
    # Extract the solution
    solution = []
    
    if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT or model.status == GRB.INTERRUPTED:
        if model.SolCount > 0:
            # Get the slide at each position
            slide_positions = {}
            for i in range(n_slides):
                for pos in range(n_slides):
                    if abs(pos_vars[(i, pos)].X - 1.0) < 1e-6:  # Check if variable is 1
                        slide_positions[pos] = i
            
            # Extract the solution in order of positions
            for pos in range(n_slides):
                if pos in slide_positions:
                    i = slide_positions[pos]
                    solution.append(all_slides[i]['photos'])
        else:
            print("No solution found within time limit.")
            # Fall back to a simple heuristic solution - just order the slides
            for slide in all_slides:
                solution.append(slide['photos'])
    else:
        print(f"Optimization failed with status: {model.status}")
        # Fall back to a simple solution
        for slide in all_slides:
            solution.append(slide['photos'])
    
    return solution


def write_solution(solution, output_file="slideshow.sol"):
    """Write the solution to the output file."""
    with open(output_file, 'w') as f:
        # First line: number of slides
        f.write(f"{len(solution)}\n")
        
        # Each subsequent line: photo ID(s) in the slide
        for slide in solution:
            f.write(" ".join(map(str, slide)) + "\n")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Photo Slideshow Optimizer")
    parser.add_argument("input_file", help="Path to the input dataset file")
    parser.add_argument("--output", default="slideshow.sol", help="Path to output solution file")
    
    args = parser.parse_args()
    
    print(f"Processing input file: {args.input_file}")
    photos = parse_input(args.input_file)
    print(f"Read {len(photos)} photos")
    
    h_slides, v_photos = create_slides_from_photos(photos)
    print(f"Created {len(h_slides)} horizontal slides and found {len(v_photos)} vertical photos")
    
    solution = optimize_slideshow(h_slides, v_photos)
    print(f"Generated solution with {len(solution)} slides")
    
    write_solution(solution, args.output)
    print(f"Solution written to {args.output}")


if __name__ == "__main__":
    main()