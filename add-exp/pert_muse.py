import json

# --- Load JSON file ---
with open("./newData/forget_prevKnow40_cleaned_Qwen.json", "r", encoding="utf-8") as f:
    json_data1 = json.load(f)

with open("./newData/forget_prevKnow40_cleaned_Phi.json", "r", encoding="utf-8") as f:
    json_data2 = json.load(f)

with open("./newData/forget_prevKnow40_cleaned.json", "r", encoding="utf-8") as f:
    json_data3 = json.load(f)

newjson = []
for d in range(len(json_data1)):
    entry = {"q_org": json_data1[d]['q_org']} 
    
    for j in range(5):
        entry.update({f"qqwen{j+1}": json_data1[d][f"qqwen{j+1}"]})
        entry.update({f"qphi{j+1}": json_data2[d][f"qphi{j+1}"]})
        entry.update({f"qmistral{j+1}" :json_data3[d][f"q{j+1}"]})
    entry.update({"answer": json_data1[d]["answer"]}) 
    newjson.append(entry)  # Append to list
    print(newjson)

with open("./newData/forget_prevKnow40_cleaned_all_paraphrases.json", "w") as file:
    json.dump(newjson, file)



