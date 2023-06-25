import json

# Load the data
valid_list = [2, 3, 4, 5, 7, 10, 11, 17, 22, 23, 27, 34, 35]

for num in valid_list:
    file_name = f"Transcription{num}.txt"
    with open(file_name, 'r') as f:
        data = json.load(f)

    output = []
    for entry in data["segments"]:
        for word in entry['words']:
            output.append([word['text'], word['start'], word['end']])

    # Save output to a separate file for each input file
    with open(f'wordLib{num}.json', 'w') as f:
        json.dump(output, f)