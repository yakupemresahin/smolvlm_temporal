import json
from SmolVLM_video_inference import ask_question

questions = [
    "Assess whether the action in the video appears physically plausible and consistent with the normal flow of time. Explain why"
]

FPSs = [3, 8]

video_names = [
    "apple_falling",
    "apple_falling_rev",
    "door_closing",
    "door_closing_rev",
    "door_opening",
    "door_opening_rev",
    "glass_breaking",
    "glass_breaking_rev"
]

outputs = []
for video_name in video_names:
    for question in questions:
        for fps in FPSs:
            
            print("-" * 50)
            print(f"Processing {video_name} with question: '{question}' at {fps} FPS")
            
            # Call the ask_question function
            output = ask_question(video_name, question, fps=fps)
            outputs.append(output)
            
# Save the outputs to a JSON file
with open("results.json", "w") as f:
    json.dump(outputs, f, indent=4)
    print(f"Results saved to results.json")