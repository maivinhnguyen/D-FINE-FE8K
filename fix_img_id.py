import json

def get_image_Id(img_name):
    img_name = img_name.split('.png')[0]
    sceneList = ['M', 'A', 'E', 'N']
    cameraIndx = int(img_name.split('_')[0].split('camera')[1])
    sceneIndx = sceneList.index(img_name.split('_')[1])
    frameIndx = int(img_name.split('_')[2])
    imageId = int(str(cameraIndx) + str(sceneIndx) + str(frameIndx))
    return imageId

def update_json_image_ids(json_path, output_path):
    # Load the original JSON
    with open(json_path, 'r') as f:
        data = json.load(f)

    # Build a mapping from original image id to new image id
    id_mapping = {}
    for image in data.get('images', []):
        original_id = image['id']
        file_name = image['file_name']
        new_id = get_image_Id(file_name)
        image['id'] = new_id
        id_mapping[original_id] = new_id

    # Update image_id in annotations using the mapping
    for ann in data.get('annotations', []):
        if ann['image_id'] in id_mapping:
            ann['image_id'] = id_mapping[ann['image_id']]
        else:
            raise ValueError(f"Annotation references unknown image_id: {ann['image_id']}")

    # Save the updated JSON
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)

# Example usage
update_json_image_ids('val.json', 'val_fixed.json')
