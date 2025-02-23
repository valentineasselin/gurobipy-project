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


def generate_initial_pairs(v_photos, max_pairs=5000):
    pairs = []
    photo_scores = defaultdict(float)
    
    # Calculate initial scores based on tag diversity
    for photo in v_photos:
        photo_scores[photo['id']] = len(photo['tags'])
    
    # Generate pairs prioritizing photos with diverse tags
    used_photos = set()
    for i, photo_i in enumerate(v_photos):
        if photo_i['id'] in used_photos:
            continue
            
        best_matches = []
        for j, photo_j in enumerate(v_photos[i+1:], i+1):
            if photo_j['id'] in used_photos:
                continue
                
            # Score based on combined tag diversity and individual photo scores
            combined_tags = photo_i['tags'] | photo_j['tags']
            score = (len(combined_tags) + 
                    photo_scores[photo_i['id']] + 
                    photo_scores[photo_j['id']])
            
            best_matches.append((score, photo_j))
        
        # Take top N matches for this photo
        best_matches.sort(reverse=True)
        for score, photo_j in best_matches[:5]:  # Consider top 5 matches per photo
            if len(pairs) >= max_pairs:
                break
            if photo_j['id'] not in used_photos:
                pairs.append({
                    'photos': [photo_i['id'], photo_j['id']],
                    'tags': photo_i['tags'] | photo_j['tags']
                })
                used_photos.add(photo_i['id'])
                used_photos.add(photo_j['id'])
                break
    
    return pairs

def find_promising_pairs(model, v_photos, current_pairs, dual_values):
    new_pairs = []
    used_in_current = {p for pair in current_pairs for p in pair['photos']}
    
    # Sort photos by their potential contribution
    photo_potential = defaultdict(float)
    for photo in v_photos:
        if photo['id'] not in used_in_current:
            # Simple score based on tag diversity
            score = len(photo['tags'])
            if dual_values is not None:
                # Use dual values to adjust score if available
                # This is a simplified approach - could be refined based on problem specifics
                score *= (1 + abs(sum(dual_values)) / len(dual_values))
            photo_potential[photo['id']] = score
    
    sorted_photos = sorted([(v, k) for k, v in photo_potential.items()], reverse=True)
    
    # Try to generate promising new pairs
    pairs_added = 0
    max_new_pairs = 1000  # Limit number of new pairs per iteration
    
    for _, photo_i_id in sorted_photos:
        if pairs_added >= max_new_pairs:
            break
            
        photo_i = next(p for p in v_photos if p['id'] == photo_i_id)
        
        # Find potential partners for this photo
        candidates = []
        for photo_j in v_photos:
            if (photo_j['id'] != photo_i_id and 
                photo_j['id'] not in used_in_current):
                
                combined_tags = photo_i['tags'] | photo_j['tags']
                score = len(combined_tags)
                
                # Additional scoring factors could be added here
                # For example, considering tag overlap with existing slides
                
                candidates.append((score, photo_j))
        
        # Take best candidates for this photo
        candidates.sort(reverse=True)
        for score, photo_j in candidates[:3]:  # Try top 3 matches per photo
            if pairs_added >= max_new_pairs:
                break
                
            # Create new pair
            new_pair = {
                'photos': [photo_i['id'], photo_j['id']],
                'tags': photo_i['tags'] | photo_j['tags']
            }
            
            # Check if this pair would likely improve the solution
            # This could be refined based on problem specifics
            if len(new_pair['tags']) >= 3:  # Simple threshold
                new_pairs.append(new_pair)
                used_in_current.add(photo_i['id'])
                used_in_current.add(photo_j['id'])
                pairs_added += 1
                break
    
    return new_pairs

def optimize_slideshow(h_slides, v_photos):
    # Start with initial set of vertical photo pairs
    print(f"Creating initial pairs for {len(v_photos)} vertical photos...")
    v_pairs = generate_initial_pairs(v_photos)
    print(f"Created {len(v_pairs)} initial vertical photo pairs")
    
    best_solution = None
    best_objective = float('-inf')
    iteration = 0
    max_iterations = 5  # Limit number of iterations for time constraints
    
    while iteration < max_iterations:
    
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
    
    # Set Gurobi parameters for this iteration
    iteration_time = 60 if iteration == 0 else 30  # Give more time to first iteration
    model.setParam('TimeLimit', iteration_time)
    model.setParam('MIPFocus', 1)
    model.setParam('Threads', 0)
    model.setParam('OutputFlag', 1)
        
    if n_slides > 100:
        model.setParam('Heuristics', 0.8)
        model.setParam('MIPGap', 0.1)  # Relax gap for speed
    
    # Optimize the model
    model.optimize()
        
    # Check if we found a solution
    if model.SolCount > 0:
        current_objective = model.ObjVal
            
        # Extract current solution
        current_solution = []
        slide_positions = {}
        for i in range(n_slides):
            for pos in range(n_slides):
                if abs(pos_vars[(i, pos)].X - 1.0) < 1e-6:
                    slide_positions[pos] = i
        
        for pos in range(n_slides):
            if pos in slide_positions:
                i = slide_positions[pos]
                current_solution.append(all_slides[i]['photos'])
        
        # Update best solution if improved
        if current_objective > best_objective:
            best_objective = current_objective
            best_solution = current_solution
            print(f"Found improved solution with objective {best_objective}")
    
    # Get dual values and generate new promising pairs
    if iteration < max_iterations - 1:  # Skip for last iteration
        dual_values = None  # Extract relevant dual values from the model
        new_pairs = find_promising_pairs(model, v_photos, v_pairs, dual_values)
        print(f"Generated {len(new_pairs)} new pairs")
        
        # Add new pairs to the pool
        v_pairs.extend(new_pairs)
        all_slides = h_slides + v_pairs
        n_slides = len(all_slides)
    
    iteration += 1
    print(f"Completed iteration {iteration}/{max_iterations}")
    
    if best_solution is None:
        print("No solution found, using fallback approach")
        # Simple fallback: just use initial pairs in sequence
        best_solution = []
        for slide in all_slides:
            best_solution.append(slide['photos'])
    
    return best_solution


def write_solution(solution, output_file="slideshow.sol"):
    with open(output_file, 'w') as f:
        # First line: number of slides
        f.write(f"{len(solution)}\n")
        
        # Each subsequent line: photo ID(s) in the slide
        for slide in solution:
            f.write(" ".join(map(str, slide)) + "\n")


def main():
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