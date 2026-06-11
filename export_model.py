"""
Export the Random Forest model's decision trees to JSON so
the prediction can run entirely in JavaScript (no Python backend needed).
"""
import pickle, json, numpy as np

with open('api/rf_model.pkl', 'rb') as f:
    model = pickle.load(f)

def tree_to_dict(tree, feature_names):
    t = tree.tree_
    def recurse(node):
        if t.children_left[node] == -1:  # leaf
            values = t.value[node][0].tolist()
            return {"leaf": True, "class": int(np.argmax(values))}
        return {
            "leaf": False,
            "feature": int(t.feature[node]),
            "threshold": float(t.threshold[node]),
            "left": recurse(int(t.children_left[node])),
            "right": recurse(int(t.children_right[node]))
        }
    return recurse(0)

feature_names = [
    "MDVP:Fo(Hz)", "MDVP:Fhi(Hz)", "MDVP:Flo(Hz)", "MDVP:Jitter(%)",
    "MDVP:Jitter(Abs)", "MDVP:RAP", "MDVP:PPQ", "Jitter:DDP",
    "MDVP:Shimmer", "MDVP:Shimmer(dB)", "Shimmer:APQ3", "Shimmer:APQ5",
    "MDVP:APQ", "Shimmer:DDA", "NHR", "HNR",
    "RPDE", "DFA", "spread1", "spread2", "D2", "PPE"
]

trees_json = [tree_to_dict(est, feature_names) for est in model.estimators_]

output = {
    "n_trees": len(trees_json),
    "n_features": 22,
    "feature_names": feature_names,
    "trees": trees_json
}

with open('model.json', 'w') as f:
    json.dump(output, f)

print(f"Exported {len(trees_json)} trees to model.json")
print(f"File size: {round(len(json.dumps(output))/1024, 1)} KB")
