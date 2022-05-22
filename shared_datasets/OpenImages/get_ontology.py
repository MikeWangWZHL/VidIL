import csv
import json

def load_label_dict():
    # load label dict
    id_2_label = {}
    for csv_path in [
        '/shared/nas/data/m1/wangz3/Shared_Datasets/VL/OpenImages/oidv6-class-descriptions.csv',
        '/shared/nas/data/m1/wangz3/Shared_Datasets/VL/OpenImages/oidv6-attributes-description.csv']:
        with open(csv_path, newline='') as csvfile:
            file_ = csv.reader(csvfile, delimiter=',', quotechar='|')
            for row in file_:
                if not row[0].startswith('/'):
                    continue
                id_, text = row[0], row[1]
                text = text.lstrip('\'').lstrip('\"').strip()
                id_2_label[id_] = text
    return id_2_label

def get_objects():
    # csv_path = '/shared/nas/data/m1/wangz3/Shared_Datasets/VL/OpenImages/oidv6-class-descriptions.csv'
    # output_path = 'openimage_classes_all.json'
    csv_path = '/shared/nas/data/m1/wangz3/Shared_Datasets/VL/OpenImages/class-descriptions-boxable.csv'
    output_path = 'openimage_classes_600.json'
    onto_list = []
    with open(csv_path, newline='') as csvfile:
        file_ = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in file_:
            if not row[0].startswith('/'):
                continue
            text = row[1]
            text = text.lstrip('\'')
            text = text.lstrip('\"')
            text = text.strip().lower()
            if text not in onto_list:
                onto_list.append(text)
    onto_list = sorted(onto_list)

    with open(output_path, 'w') as out:
        json.dump(onto_list, out, indent=4)

def get_relation_triples():

    id_2_label = load_label_dict()

    csv_path = '/shared/nas/data/m1/wangz3/Shared_Datasets/VL/OpenImages/oidv6-relationship-triplets.csv'
    output_path = 'openimage_relation_triples.json'
    onto_list = []
    with open(csv_path, newline='') as csvfile:
        file_ = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in file_:
            if not row[0].startswith('/'):
                continue
            obj_id,sub_id,rel = row
            obj = id_2_label[obj_id].lower()
            sub = id_2_label[sub_id].lower()
            rel = rel.replace('_',' ').lower()
            text = f'{obj} {rel} {sub}' 
            if text not in onto_list:
                onto_list.append(text)
    onto_list = sorted(onto_list)

    with open(output_path, 'w') as out:
        json.dump(onto_list, out, indent=4)

if __name__ == '__main__':
    get_objects()
    get_relation_triples()
