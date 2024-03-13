import csv
from pprint import pprint


def read_csv_to_dict(path):
    # Create an empty list to store the dictionaries
    data_as_dict = []

    # Open the CSV file
    with open(path, mode='r') as file:
        # Use DictReader to read the CSV into dictionaries
        reader = csv.DictReader(file)

        # Iterate over the rows in the reader
        for row in reader:
            # Each row is already a dictionary; add it to the list
            data_as_dict.append(row)

    return data_as_dict


if __name__ == '__main__':
    d = read_csv_to_dict('../data/external/zipcodes.csv')
    result = {}
    vorige_hoofdgemeente = d[0]['Hoofdgemeente']
    vorige_hoofdgemeente_postcode = d[0]['Postcode']
    for item in d:
        if vorige_hoofdgemeente == item['Hoofdgemeente']:
            result[item['Postcode']] = vorige_hoofdgemeente_postcode
        else:
            result[item['Postcode']] = item['Postcode']
            vorige_hoofdgemeente = item['Hoofdgemeente']
            vorige_hoofdgemeente_postcode = item['Postcode']
    pprint(result)
