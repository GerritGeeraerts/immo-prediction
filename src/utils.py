from enum import Enum

import pickle


def save_model_as_pickle(model, file_path):
    """
    Saves the given model as a pickle file to the specified file path.

    Parameters:
    - model: The model to be saved.
    - file_path: The path where the model will be saved, including the file name.
    """
    with open(file_path, 'wb') as file:
        pickle.dump(model, file)


class ImmoFeature(Enum):
    id = 'ID'
    locality = 'Locality'
    postal_code = 'Postal Code'
    build_year = 'Build Year'
    facades = 'Facades'
    habitable_surface = 'Habitable Surface'
    land_surface = 'Land Surface'
    type = 'Type'
    subtype = 'Subtype'
    price = 'Price'
    sale_type = 'Sale Type'
    bedroom_count = 'Bedroom Count'
    bathroom_count = 'Bathroom Count'
    toilet_count = 'Toilet Count'
    room_count = 'Room Count'
    kitchen_surface = 'Kitchen Surface'
    kitchen = 'Kitchen'
    kitchen_type = 'Kitchen Type'
    furnished = 'Furnished'
    openfire = 'Openfire'
    fireplace_count = 'Fireplace Count'
    terrace = 'Terrace'
    terrace_surface = 'Terrace Surface'
    terrace_orientation = 'Terrace Orientation'
    garden_exists = 'Garden Exists'
    garden_surface = 'Garden Surface'
    garden_orientation = 'Garden Orientation'
    swimming_pool = 'Swimming Pool'
    state_of_building = 'State of Building'
    living_surface = 'Living Surface'
    epc = 'EPC'
    cadastral_income = 'Cadastral Income'
    has_starting_price = 'Has starting Price'
    transaction_subtype = 'Transaction Subtype'
    heating_type = 'Heating Type'
    is_holiday_property = 'Is Holiday Property'
    gas_water_electricity = 'Gas Water Electricity'
    sewer = 'Sewer'
    sea_view = 'Sea view'
    parking_count_inside = 'Parking count inside'
    parking_count_outside = 'Parking count outside'
    url = 'url'
    longitude = 'Longitude'
    latitude = 'Latitude'

    # calculated features
    # 'VILLA', 'HOUSE', 'APARTMENT', 'MANSION', 'PENTHOUSE', 'TOWN_HOUSE', 'GROUND_FLOOR', 'FLAT_STUDIO', 'DUPLEX',
    apartment = 'Subtype_APARTMENT'
    house = 'Subtype_HOUSE'
    villa = 'Subtype_VILLA'
    mansion = 'Subtype_MANSION'
    penthouse = 'Subtype_PENTHOUSE'
    town_house = 'Subtype_TOWN_HOUSE'
    ground_floor = 'Subtype_GROUND_FLOOR'
    flat_studio = 'Subtype_FLAT_STUDIO'
    duplex = 'Subtype_DUPLEX'
