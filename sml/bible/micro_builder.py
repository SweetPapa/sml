"""Micro-PoC Bible builder with ~130 hand-crafted concepts."""

import sqlite3
from sml.bible.schema import create_bible_db


def build_micro_bible(db_path: str) -> None:
    """Build a micro Bible with ~130 concepts and ~190 relations for PoC testing."""
    create_bible_db(db_path)

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    # ── Concepts ─────────────────────────────────────────────────────────
    # (id, uri, surface_text, anchor_token, domain, category, subcategory, specificity, definition)

    concepts = [
        # Animals (domain=1 physical, category=1 living, subcategory=2 animal)
        (1001, "/c/en/dog", "dog", "dog_1001", 1, 1, 2, 1, "A domesticated carnivorous mammal"),
        (1002, "/c/en/cat", "cat", "cat_1002", 1, 1, 2, 1, "A small domesticated feline"),
        (1003, "/c/en/bird", "bird", "bird_1003", 1, 1, 2, 2, "A warm-blooded egg-laying vertebrate with feathers"),
        (1004, "/c/en/fish", "fish", "fish_1004", 1, 1, 2, 4, "A limbless cold-blooded aquatic vertebrate"),
        (1005, "/c/en/penguin", "penguin", "penguin_1005", 1, 1, 2, 3, "A flightless seabird of the southern hemisphere"),
        (1006, "/c/en/snake", "snake", "snake_1006", 1, 1, 2, 3, "A long limbless reptile"),
        (1007, "/c/en/elephant", "elephant", "elephant_1007", 1, 1, 2, 3, "A very large herbivorous mammal with a trunk"),
        (1008, "/c/en/mouse", "mouse", "mouse_1008", 1, 1, 2, 3, "A small rodent with a pointed snout"),

        # Animal taxonomy (domain=1, category=1 living, subcategory=2 animal)
        (1009, "/c/en/animal", "animal", "animal_1009", 1, 1, 2, 0, "A living organism that feeds on organic matter"),
        (1010, "/c/en/reptile", "reptile", "reptile_1010", 1, 1, 2, 0, "A cold-blooded vertebrate with scales"),
        (1011, "/c/en/mammal", "mammal", "mammal_1011", 1, 1, 2, 0, "A warm-blooded vertebrate that nurses its young"),
        (1012, "/c/en/pet", "pet", "pet_1012", 1, 1, 2, 0, "An animal kept for companionship"),

        # Body parts (domain=1, category=2 object)
        (1201, "/c/en/leg", "leg", "leg_1201", 1, 2, 0, 0, "A limb used for standing and walking"),
        (1202, "/c/en/tail", "tail", "tail_1202", 1, 2, 0, 0, "A flexible extension at the rear of an animal"),
        (1203, "/c/en/ear", "ear", "ear_1203", 1, 2, 0, 0, "An organ of hearing"),
        (1204, "/c/en/wing", "wing", "wing_1204", 1, 2, 0, 0, "A limb used for flying"),
        (1205, "/c/en/fur", "fur", "fur_1205", 1, 2, 0, 0, "Soft thick hair covering an animal"),
        (1206, "/c/en/leaf", "leaf", "leaf_1206", 1, 2, 0, 0, "A flat green structure on a plant"),
        (1207, "/c/en/trunk", "trunk", "trunk_1207", 1, 2, 0, 0, "The elongated nose of an elephant"),
        (1208, "/c/en/scale", "scale", "scale_1208", 1, 2, 0, 0, "A small rigid plate on the skin of fish or reptiles"),

        # Objects (domain=1, category=2 object)
        (2001, "/c/en/ball", "ball", "ball_2001", 1, 2, 0, 0, "A solid or hollow spherical object"),
        (2002, "/c/en/mat", "mat", "mat_2002", 1, 2, 0, 0, "A piece of material used as a floor covering"),
        (2003, "/c/en/car", "car", "car_2003", 1, 2, 0, 0, "A four-wheeled motor vehicle"),
        (2004, "/c/en/house", "house", "house_2004", 1, 2, 0, 0, "A building for human habitation"),
        (2005, "/c/en/book", "book", "book_2005", 1, 2, 0, 0, "A written or printed work of pages"),
        (2006, "/c/en/table", "table", "table_2006", 1, 2, 0, 0, "A piece of furniture with a flat top"),
        (2007, "/c/en/chair", "chair", "chair_2007", 1, 2, 0, 0, "A seat for one person with a back"),

        # People (domain=1, category=1 living, subcategory=1 human)
        (1101, "/c/en/person", "person", "person_1101", 1, 1, 1, 0, "A human being"),
        (1102, "/c/en/child", "child", "child_1102", 1, 1, 1, 0, "A young human being"),

        # Colors (domain=4 property, category=1 color)
        (3001, "/c/en/red", "red", "red_3001", 4, 1, 0, 0, "The color of blood or fire"),
        (3002, "/c/en/blue", "blue", "blue_3002", 4, 1, 0, 0, "The color of the sky on a clear day"),
        (3003, "/c/en/green", "green", "green_3003", 4, 1, 0, 0, "The color of grass and leaves"),
        (3004, "/c/en/yellow", "yellow", "yellow_3004", 4, 1, 0, 0, "The color of sunflowers and lemons"),
        (3005, "/c/en/brown", "brown", "brown_3005", 4, 1, 0, 0, "The color of earth or wood"),
        (3006, "/c/en/white", "white", "white_3006", 4, 1, 0, 0, "The lightest color, the color of snow"),
        (3007, "/c/en/black", "black", "black_3007", 4, 1, 0, 0, "The darkest color, the absence of light"),
        (3008, "/c/en/orange", "orange", "orange_3008", 4, 1, 0, 0, "The color between red and yellow"),

        # Sizes (domain=4 property, category=2 size)
        (3101, "/c/en/big", "big", "big_3101", 4, 2, 0, 0, "Of considerable size or extent"),
        (3102, "/c/en/small", "small", "small_3102", 4, 2, 0, 0, "Of limited size or extent"),

        # Places (domain=1, category=4 place)
        (4001, "/c/en/park", "park", "park_4001", 1, 4, 0, 0, "A public area of land for recreation"),
        (4002, "/c/en/kitchen", "kitchen", "kitchen_4002", 1, 4, 0, 0, "A room where food is prepared"),
        (4003, "/c/en/school", "school", "school_4003", 1, 4, 0, 0, "An institution for educating children"),
        (4004, "/c/en/ocean", "ocean", "ocean_4004", 1, 4, 0, 0, "A vast body of salt water"),
        (4005, "/c/en/ground", "ground", "ground_4005", 1, 4, 0, 0, "The solid surface of the earth"),
        (4006, "/c/en/land", "land", "land_4006", 1, 4, 0, 0, "The solid part of the earth's surface"),
        (4007, "/c/en/lake", "lake", "lake_4007", 1, 4, 0, 0, "A large body of fresh water"),
        (4008, "/c/en/pond", "pond", "pond_4008", 1, 4, 0, 0, "A small body of still water"),
        (4009, "/c/en/nest", "nest", "nest_4009", 1, 4, 0, 0, "A structure built by birds for eggs"),
        (4010, "/c/en/forest", "forest", "forest_4010", 1, 4, 0, 0, "A large area covered with trees"),
        (4011, "/c/en/river", "river", "river_4011", 1, 4, 0, 0, "A large natural stream of flowing water"),
        (4012, "/c/en/glass", "glass", "glass_4012", 1, 2, 0, 0, "A drinking vessel made of glass"),

        # Actions/Verbs (domain=3 action, category=0)
        (5001, "/c/en/sit", "sit", "sit_5001", 3, 0, 0, 0, "To be in a position with weight on buttocks"),
        (5002, "/c/en/run", "run", "run_5002", 3, 0, 0, 0, "To move at a speed faster than walking"),
        (5003, "/c/en/eat", "eat", "eat_5003", 3, 0, 0, 0, "To put food in the mouth and swallow"),
        (5004, "/c/en/fly", "fly", "fly_5004", 3, 0, 0, 0, "To move through the air with wings"),
        (5005, "/c/en/swim", "swim", "swim_5005", 3, 0, 0, 0, "To move through water using the body"),
        (5006, "/c/en/bark", "bark", "bark_5006", 3, 0, 0, 0, "To make a sharp explosive cry"),
        (5007, "/c/en/purr", "purr", "purr_5007", 3, 0, 0, 0, "To make a low continuous vibrating sound"),
        (5008, "/c/en/sleep", "sleep", "sleep_5008", 3, 0, 0, 0, "A state of rest with reduced consciousness"),
        (5009, "/c/en/play", "play", "play_5009", 3, 0, 0, 0, "To engage in activity for enjoyment"),
        (5010, "/c/en/walk", "walk", "walk_5010", 3, 0, 0, 0, "To move at a regular pace by lifting each foot in turn"),
        (5011, "/c/en/hear", "hear", "hear_5011", 3, 0, 0, 0, "To perceive sound with the ears"),
        (5012, "/c/en/climb", "climb", "climb_5012", 3, 0, 0, 0, "To go or come up a slope or incline"),
        (5013, "/c/en/chase", "chase", "chase_5013", 3, 0, 0, 0, "To pursue in order to catch"),
        (5014, "/c/en/meow", "meow", "meow_5014", 3, 0, 0, 0, "The cry of a cat"),
        (5015, "/c/en/sing", "sing", "sing_5015", 3, 0, 0, 0, "To make musical sounds with the voice"),
        (5016, "/c/en/grow", "grow", "grow_5016", 3, 0, 0, 0, "To increase in size or develop"),
        (5017, "/c/en/melt", "melt", "melt_5017", 3, 0, 0, 0, "To change from solid to liquid by heating"),
        (5018, "/c/en/freeze", "freeze", "freeze_5018", 3, 0, 0, 0, "To change from liquid to solid by cooling"),
        (5019, "/c/en/build", "build", "build_5019", 3, 0, 0, 0, "To construct by putting parts together"),
        (5020, "/c/en/wag", "wag", "wag_5020", 3, 0, 0, 0, "To move rapidly back and forth"),
        (5021, "/c/en/drink", "drink", "drink_5021", 3, 0, 0, 0, "To take liquid into the mouth and swallow"),
        (5022, "/c/en/read", "read", "read_5022", 3, 0, 0, 0, "To look at and understand written words"),
        (5023, "/c/en/cook", "cook", "cook_5023", 3, 0, 0, 0, "To prepare food by heating"),
        (5024, "/c/en/learn", "learn", "learn_5024", 3, 0, 0, 0, "To gain knowledge or skill"),

        # Properties - temperature (domain=4, category=3)
        (3201, "/c/en/hot", "hot", "hot_3201", 4, 3, 0, 0, "Having a high temperature"),
        (3202, "/c/en/cold", "cold", "cold_3202", 4, 3, 0, 0, "Having a low temperature"),
        (3419, "/c/en/warm", "warm", "warm_3419", 4, 3, 0, 0, "Having moderate heat"),

        # Properties - speed (domain=4, category=4)
        (3203, "/c/en/fast", "fast", "fast_3203", 4, 4, 0, 0, "Moving at high speed"),
        (3204, "/c/en/slow", "slow", "slow_3204", 4, 4, 0, 0, "Moving at low speed"),

        # Properties - age (domain=4, category=5)
        (3205, "/c/en/old", "old", "old_3205", 4, 5, 0, 0, "Having lived for many years"),
        (3206, "/c/en/new", "new", "new_3206", 4, 5, 0, 0, "Recently made or discovered"),
        (3424, "/c/en/young", "young", "young_3424", 4, 5, 0, 0, "Having lived for only a short time"),

        # Properties - luminosity (domain=4, category=6)
        (3301, "/c/en/bright", "bright", "bright_3301", 4, 6, 0, 0, "Giving out or reflecting much light"),
        (3302, "/c/en/dark", "dark", "dark_3302", 4, 6, 0, 0, "With little or no light"),

        # Properties - weight (domain=4, category=7)
        (3401, "/c/en/heavy", "heavy", "heavy_3401", 4, 7, 0, 0, "Of great weight; difficult to lift"),
        (3402, "/c/en/light", "light", "light_3402", 4, 7, 0, 0, "Of little weight; not heavy"),

        # Properties - texture (domain=4, category=8)
        (3403, "/c/en/soft", "soft", "soft_3403", 4, 8, 0, 0, "Easy to compress or bend; not hard"),
        (3404, "/c/en/hard", "hard", "hard_3404", 4, 8, 0, 0, "Solid and rigid; not easily compressed"),

        # Properties - taste (domain=4, category=9)
        (3405, "/c/en/sweet", "sweet", "sweet_3405", 4, 9, 0, 0, "Having the taste of sugar"),
        (3406, "/c/en/salty", "salty", "salty_3406", 4, 9, 0, 0, "Having the taste of salt"),

        # Properties - shape (domain=4, category=10)
        (3407, "/c/en/round", "round", "round_3407", 4, 10, 0, 0, "Shaped like a circle or sphere"),
        (3408, "/c/en/long", "long", "long_3408", 4, 10, 0, 0, "Of great length or duration"),
        (3409, "/c/en/tall", "tall", "tall_3409", 4, 2, 0, 0, "Of great vertical extent"),
        (3410, "/c/en/deep", "deep", "deep_3410", 4, 2, 0, 0, "Extending far down from the surface"),

        # Properties - sound (domain=4, category=11)
        (3411, "/c/en/quiet", "quiet", "quiet_3411", 4, 11, 0, 0, "Making little or no noise"),
        (3412, "/c/en/loud", "loud", "loud_3412", 4, 11, 0, 0, "Making a lot of noise"),

        # Properties - clarity (domain=4, category=12)
        (3413, "/c/en/clear", "clear", "clear_3413", 4, 12, 0, 0, "Transparent; easy to see through"),

        # Properties - personality (domain=4, category=13)
        (3414, "/c/en/friendly", "friendly", "friendly_3414", 4, 13, 0, 0, "Kind and pleasant"),
        (3415, "/c/en/loyal", "loyal", "loyal_3415", 4, 13, 0, 0, "Giving firm and constant support"),
        (3416, "/c/en/independent", "independent", "independent_3416", 4, 13, 0, 0, "Not depending on others"),
        (3417, "/c/en/cute", "cute", "cute_3417", 4, 13, 0, 0, "Attractive in a pretty way"),
        (3418, "/c/en/quick", "quick", "quick_3418", 4, 13, 0, 0, "Moving fast or doing something in a short time"),

        # Properties - misc (domain=4)
        (3420, "/c/en/healthy", "healthy", "healthy_3420", 4, 0, 0, 0, "In good physical condition"),
        (3421, "/c/en/calm", "calm", "calm_3421", 4, 0, 0, 0, "Not showing or feeling agitation"),
        (3422, "/c/en/dangerous", "dangerous", "dangerous_3422", 4, 0, 0, 0, "Able to cause harm or injury"),
        (3423, "/c/en/useful", "useful", "useful_3423", 4, 0, 0, 0, "Able to be used for a practical purpose"),

        # Food / Substances (domain=1, category=3 substance)
        (6001, "/c/en/apple", "apple", "apple_6001", 1, 3, 0, 0, "A round fruit with red or green skin"),
        (6002, "/c/en/bread", "bread", "bread_6002", 1, 3, 0, 0, "Food made from flour, water, and yeast"),
        (6003, "/c/en/water", "water", "water_6003", 1, 3, 0, 0, "A transparent liquid essential for life"),
        (6004, "/c/en/milk", "milk", "milk_6004", 1, 3, 0, 0, "White liquid produced by mammals"),
        (6005, "/c/en/ice", "ice", "ice_6005", 1, 3, 0, 0, "Frozen water; a solid form of water"),
        (6006, "/c/en/cheese", "cheese", "cheese_6006", 1, 3, 0, 0, "A food made from pressed milk curds"),
        (6007, "/c/en/food", "food", "food_6007", 1, 3, 0, 0, "Any substance consumed for nutrition"),
        (6008, "/c/en/plant", "plant", "plant_6008", 1, 1, 3, 0, "A living organism that grows in the earth"),
        (6009, "/c/en/fruit", "fruit", "fruit_6009", 1, 3, 0, 0, "The sweet product of a tree or plant"),

        # Abstract (domain=2 abstract)
        (7001, "/c/en/love", "love", "love_7001", 2, 0, 0, 0, "An intense feeling of deep affection"),
        (7002, "/c/en/fear", "fear", "fear_7002", 2, 0, 0, 0, "An unpleasant emotion caused by threat"),
        (7003, "/c/en/knowledge", "knowledge", "knowledge_7003", 2, 0, 0, 0, "Facts or information acquired through experience"),

        # Celestial / Natural (domain=1, category=5)
        (8001, "/c/en/sun", "sun", "sun_8001", 1, 5, 0, 0, "The star at the center of the solar system"),
        (8002, "/c/en/sky", "sky", "sky_8002", 1, 5, 0, 0, "The region of the atmosphere seen from earth"),
        (8003, "/c/en/night", "night", "night_8003", 1, 5, 0, 0, "The period of darkness between sunset and sunrise"),
        (8004, "/c/en/fire", "fire", "fire_8004", 1, 5, 0, 0, "Combustion producing heat, light, and flame"),
        (8005, "/c/en/snow", "snow", "snow_8005", 1, 5, 0, 0, "Frozen atmospheric water vapor falling as white flakes"),
        (8006, "/c/en/morning", "morning", "morning_8006", 1, 5, 0, 0, "The early part of the day"),
        (8007, "/c/en/evening", "evening", "evening_8007", 1, 5, 0, 0, "The later part of the day"),
        (8008, "/c/en/star", "star", "star_8008", 1, 5, 0, 0, "A luminous celestial body"),
        (8009, "/c/en/rain", "rain", "rain_8009", 1, 5, 0, 0, "Water falling from clouds in drops"),

        # Plants / Nature (domain=1, category=1 living, subcategory=3 plant)
        (9001, "/c/en/grass", "grass", "grass_9001", 1, 1, 3, 0, "Green vegetation covering the ground"),
        (9002, "/c/en/tree", "tree", "tree_9002", 1, 1, 3, 0, "A tall perennial plant with a woody trunk"),
    ]

    cur.executemany(
        "INSERT OR IGNORE INTO concepts "
        "(id, uri, surface_text, anchor_token, domain, category, subcategory, specificity, definition) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        concepts,
    )

    # ── Relations ────────────────────────────────────────────────────────
    # (source_id, target_id, relation_type_id, weight)

    relations = [
        # ── IsA (1) — Proper taxonomy ────────────────────────────────────
        (1001, 1009, 1, 0.99),   # dog IsA animal
        (1001, 1011, 1, 0.95),   # dog IsA mammal
        (1001, 1012, 1, 0.90),   # dog IsA pet
        (1002, 1009, 1, 0.99),   # cat IsA animal
        (1002, 1011, 1, 0.95),   # cat IsA mammal
        (1002, 1012, 1, 0.90),   # cat IsA pet
        (1003, 1009, 1, 0.99),   # bird IsA animal
        (1004, 1009, 1, 0.99),   # fish IsA animal
        (1005, 1003, 1, 0.95),   # penguin IsA bird
        (1005, 1009, 1, 0.99),   # penguin IsA animal
        (1006, 1010, 1, 0.95),   # snake IsA reptile
        (1006, 1009, 1, 0.99),   # snake IsA animal
        (1007, 1011, 1, 0.95),   # elephant IsA mammal
        (1007, 1009, 1, 0.99),   # elephant IsA animal
        (1008, 1011, 1, 0.95),   # mouse IsA mammal
        (1008, 1009, 1, 0.99),   # mouse IsA animal
        (8001, 8008, 1, 0.95),   # sun IsA star
        (6001, 6009, 1, 0.95),   # apple IsA fruit
        (9001, 6008, 1, 0.90),   # grass IsA plant
        (9002, 6008, 1, 0.90),   # tree IsA plant

        # ── HasA (3) — Body parts ────────────────────────────────────────
        (1001, 1201, 3, 0.95),   # dog HasA leg
        (1001, 1202, 3, 0.95),   # dog HasA tail
        (1001, 1203, 3, 0.90),   # dog HasA ear
        (1001, 1205, 3, 0.90),   # dog HasA fur
        (1002, 1201, 3, 0.95),   # cat HasA leg
        (1002, 1202, 3, 0.95),   # cat HasA tail
        (1002, 1205, 3, 0.90),   # cat HasA fur
        (1003, 1204, 3, 0.95),   # bird HasA wing
        (1003, 1201, 3, 0.90),   # bird HasA leg
        (1004, 1208, 3, 0.90),   # fish HasA scale
        (1005, 1204, 3, 0.90),   # penguin HasA wing
        (1006, 1208, 3, 0.90),   # snake HasA scale
        (1007, 1201, 3, 0.95),   # elephant HasA leg
        (1007, 1207, 3, 0.95),   # elephant HasA trunk
        (1007, 1203, 3, 0.90),   # elephant HasA ear
        (1008, 1202, 3, 0.90),   # mouse HasA tail
        (9002, 1206, 3, 0.90),   # tree HasA leaf

        # ── CapableOf (5) — Animal/person capabilities ───────────────────
        (1001, 5006, 5, 0.95),   # dog CapableOf bark
        (1001, 5002, 5, 0.90),   # dog CapableOf run
        (1001, 5005, 5, 0.70),   # dog CapableOf swim
        (1001, 5008, 5, 0.95),   # dog CapableOf sleep
        (1001, 5009, 5, 0.90),   # dog CapableOf play
        (1001, 5010, 5, 0.95),   # dog CapableOf walk
        (1001, 5013, 5, 0.85),   # dog CapableOf chase
        (1001, 5011, 5, 0.85),   # dog CapableOf hear
        (1001, 5003, 5, 0.95),   # dog CapableOf eat
        (1001, 5020, 5, 0.90),   # dog CapableOf wag
        (1002, 5007, 5, 0.95),   # cat CapableOf purr
        (1002, 5002, 5, 0.85),   # cat CapableOf run
        (1002, 5008, 5, 0.95),   # cat CapableOf sleep
        (1002, 5012, 5, 0.85),   # cat CapableOf climb
        (1002, 5014, 5, 0.90),   # cat CapableOf meow
        (1002, 5013, 5, 0.80),   # cat CapableOf chase
        (1002, 5010, 5, 0.90),   # cat CapableOf walk
        (1002, 5003, 5, 0.95),   # cat CapableOf eat
        (1003, 5004, 5, 0.95),   # bird CapableOf fly
        (1003, 5015, 5, 0.85),   # bird CapableOf sing
        (1003, 5019, 5, 0.80),   # bird CapableOf build (nests)
        (1003, 5010, 5, 0.85),   # bird CapableOf walk
        (1004, 5005, 5, 0.98),   # fish CapableOf swim
        (1004, 5003, 5, 0.90),   # fish CapableOf eat
        (1005, 5005, 5, 0.95),   # penguin CapableOf swim
        (1005, 5010, 5, 0.90),   # penguin CapableOf walk
        (1005, 5003, 5, 0.90),   # penguin CapableOf eat
        (1006, 5005, 5, 0.70),   # snake CapableOf swim
        (1006, 5003, 5, 0.90),   # snake CapableOf eat
        (1007, 5010, 5, 0.95),   # elephant CapableOf walk
        (1007, 5005, 5, 0.60),   # elephant CapableOf swim
        (1007, 5003, 5, 0.95),   # elephant CapableOf eat
        (1007, 5021, 5, 0.90),   # elephant CapableOf drink
        (1008, 5002, 5, 0.80),   # mouse CapableOf run
        (1008, 5012, 5, 0.75),   # mouse CapableOf climb
        (1008, 5003, 5, 0.90),   # mouse CapableOf eat
        (1008, 5005, 5, 0.60),   # mouse CapableOf swim
        (1101, 5022, 5, 0.95),   # person CapableOf read
        (1101, 5002, 5, 0.90),   # person CapableOf run
        (1101, 5010, 5, 0.95),   # person CapableOf walk
        (1101, 5003, 5, 0.95),   # person CapableOf eat
        (1101, 5021, 5, 0.95),   # person CapableOf drink
        (1101, 5023, 5, 0.90),   # person CapableOf cook
        (1102, 5009, 5, 0.95),   # child CapableOf play
        (1102, 5024, 5, 0.90),   # child CapableOf learn
        (1102, 5022, 5, 0.85),   # child CapableOf read

        # ── AtLocation (6) — Habitat/location ────────────────────────────
        (1001, 4001, 6, 0.80),   # dog AtLocation park
        (1001, 2004, 6, 0.85),   # dog AtLocation house
        (1002, 2004, 6, 0.90),   # cat AtLocation house
        (1002, 4002, 6, 0.70),   # cat AtLocation kitchen
        (1003, 4001, 6, 0.75),   # bird AtLocation park
        (1003, 8002, 6, 0.85),   # bird AtLocation sky
        (1003, 9002, 6, 0.80),   # bird AtLocation tree
        (1003, 4009, 6, 0.85),   # bird AtLocation nest
        (1004, 6003, 6, 0.95),   # fish AtLocation water
        (1004, 4004, 6, 0.90),   # fish AtLocation ocean
        (1004, 4008, 6, 0.85),   # fish AtLocation pond
        (1004, 4007, 6, 0.85),   # fish AtLocation lake
        (1004, 4011, 6, 0.80),   # fish AtLocation river
        (1102, 4003, 6, 0.90),   # child AtLocation school
        (2005, 4003, 6, 0.85),   # book AtLocation school
        (2006, 4002, 6, 0.80),   # table AtLocation kitchen
        (2007, 4002, 6, 0.75),   # chair AtLocation kitchen
        (1005, 4004, 6, 0.85),   # penguin AtLocation ocean
        (1005, 6005, 6, 0.80),   # penguin AtLocation ice
        (1006, 4005, 6, 0.75),   # snake AtLocation ground
        (1006, 4010, 6, 0.80),   # snake AtLocation forest
        (1007, 4001, 6, 0.50),   # elephant AtLocation park (zoo/safari)
        (1007, 4010, 6, 0.80),   # elephant AtLocation forest
        (1008, 2004, 6, 0.75),   # mouse AtLocation house
        (1008, 4005, 6, 0.70),   # mouse AtLocation ground
        (9001, 4005, 6, 0.90),   # grass AtLocation ground
        (9001, 4001, 6, 0.85),   # grass AtLocation park
        (6001, 9002, 6, 0.80),   # apple AtLocation tree
        (1101, 2004, 6, 0.90),   # person AtLocation house
        (6002, 4002, 6, 0.80),   # bread AtLocation kitchen

        # ── HasProperty (4) — Properties ─────────────────────────────────
        (1001, 3005, 4, 0.70),   # dog HasProperty brown
        (1001, 3203, 4, 0.75),   # dog HasProperty fast
        (1001, 3414, 4, 0.85),   # dog HasProperty friendly
        (1001, 3415, 4, 0.85),   # dog HasProperty loyal
        (1002, 3102, 4, 0.70),   # cat HasProperty small
        (1002, 3416, 4, 0.80),   # cat HasProperty independent
        (1002, 3418, 4, 0.75),   # cat HasProperty quick
        (1003, 3402, 4, 0.80),   # bird HasProperty light
        (1005, 3007, 4, 0.70),   # penguin HasProperty black
        (1005, 3006, 4, 0.70),   # penguin HasProperty white
        (1005, 3417, 4, 0.75),   # penguin HasProperty cute
        (1006, 3408, 4, 0.85),   # snake HasProperty long
        (1007, 3101, 4, 0.95),   # elephant HasProperty big
        (1007, 3401, 4, 0.95),   # elephant HasProperty heavy
        (1007, 3204, 4, 0.80),   # elephant HasProperty slow
        (1008, 3102, 4, 0.90),   # mouse HasProperty small
        (1008, 3402, 4, 0.85),   # mouse HasProperty light
        (1008, 3411, 4, 0.75),   # mouse HasProperty quiet
        (1008, 3203, 4, 0.70),   # mouse HasProperty fast
        (6001, 3001, 4, 0.80),   # apple HasProperty red
        (6001, 3003, 4, 0.70),   # apple HasProperty green
        (6001, 3405, 4, 0.85),   # apple HasProperty sweet
        (6001, 3407, 4, 0.85),   # apple HasProperty round
        (6002, 3403, 4, 0.80),   # bread HasProperty soft
        (6003, 3202, 4, 0.60),   # water HasProperty cold
        (6003, 3413, 4, 0.85),   # water HasProperty clear
        (6004, 3006, 4, 0.85),   # milk HasProperty white
        (6004, 3420, 4, 0.75),   # milk HasProperty healthy
        (6005, 3202, 4, 0.98),   # ice HasProperty cold
        (6005, 3006, 4, 0.80),   # ice HasProperty white
        (6005, 3404, 4, 0.85),   # ice HasProperty hard
        (8001, 3004, 4, 0.95),   # sun HasProperty yellow
        (8001, 3201, 4, 0.98),   # sun HasProperty hot
        (8001, 3301, 4, 0.95),   # sun HasProperty bright
        (8002, 3002, 4, 0.95),   # sky HasProperty blue
        (8003, 3302, 4, 0.95),   # night HasProperty dark
        (8003, 3411, 4, 0.80),   # night HasProperty quiet
        (8004, 3001, 4, 0.90),   # fire HasProperty red
        (8004, 3201, 4, 0.98),   # fire HasProperty hot
        (8004, 3301, 4, 0.90),   # fire HasProperty bright
        (8004, 3422, 4, 0.90),   # fire HasProperty dangerous
        (8005, 3006, 4, 0.98),   # snow HasProperty white
        (8005, 3202, 4, 0.95),   # snow HasProperty cold
        (8005, 3403, 4, 0.80),   # snow HasProperty soft
        (9001, 3003, 4, 0.95),   # grass HasProperty green
        (9001, 3403, 4, 0.75),   # grass HasProperty soft
        (9002, 3003, 4, 0.80),   # tree HasProperty green
        (9002, 3101, 4, 0.70),   # tree HasProperty big
        (9002, 3409, 4, 0.80),   # tree HasProperty tall
        (4004, 3002, 4, 0.85),   # ocean HasProperty blue
        (4004, 3410, 4, 0.85),   # ocean HasProperty deep
        (4004, 3406, 4, 0.80),   # ocean HasProperty salty
        (4004, 3101, 4, 0.80),   # ocean HasProperty big
        (4004, 3421, 4, 0.70),   # ocean HasProperty calm
        (4001, 3003, 4, 0.80),   # park HasProperty green
        (2001, 3407, 4, 0.90),   # ball HasProperty round
        (2005, 3423, 4, 0.80),   # book HasProperty useful

        # ── Antonym (22) — Opposites (bidirectional) ─────────────────────
        (3201, 3202, 22, 0.95),  # hot Antonym cold
        (3202, 3201, 22, 0.95),  # cold Antonym hot
        (3101, 3102, 22, 0.95),  # big Antonym small
        (3102, 3101, 22, 0.95),  # small Antonym big
        (3203, 3204, 22, 0.95),  # fast Antonym slow
        (3204, 3203, 22, 0.95),  # slow Antonym fast
        (3302, 3301, 22, 0.95),  # dark Antonym bright
        (3301, 3302, 22, 0.95),  # bright Antonym dark
        (3205, 3424, 22, 0.90),  # old Antonym young
        (3424, 3205, 22, 0.90),  # young Antonym old
        (3401, 3402, 22, 0.95),  # heavy Antonym light
        (3402, 3401, 22, 0.95),  # light Antonym heavy
        (3403, 3404, 22, 0.95),  # soft Antonym hard
        (3404, 3403, 22, 0.95),  # hard Antonym soft
        (3411, 3412, 22, 0.95),  # quiet Antonym loud
        (3412, 3411, 22, 0.95),  # loud Antonym quiet

        # ── UsedFor (12) ─────────────────────────────────────────────────
        (2001, 5009, 12, 0.90),  # ball UsedFor play
        (2005, 7003, 12, 0.85),  # book UsedFor knowledge
        (2003, 5002, 12, 0.60),  # car UsedFor travel (approximate)
        (2006, 5003, 12, 0.70),  # table UsedFor eat
        (2007, 5001, 12, 0.90),  # chair UsedFor sit
        (2004, 5008, 12, 0.80),  # house UsedFor sleep
        (4003, 5024, 12, 0.90),  # school UsedFor learn
        (4002, 5023, 12, 0.85),  # kitchen UsedFor cook
        (4001, 5009, 12, 0.85),  # park UsedFor play
        (6003, 5021, 12, 0.90),  # water UsedFor drink

        # ── MadeOf (16) ──────────────────────────────────────────────────
        (6002, 6003, 16, 0.60),  # bread MadeOf water (partially)
        (6005, 6003, 16, 0.95),  # ice MadeOf water
        (8005, 6003, 16, 0.90),  # snow MadeOf water

        # ── Causes (7) ───────────────────────────────────────────────────
        (7002, 5002, 7, 0.60),   # fear Causes run
        (8004, 7002, 7, 0.50),   # fire Causes fear
        (8005, 3202, 7, 0.70),   # snow Causes cold
        (3201, 5017, 7, 0.85),   # hot Causes melt
        (3202, 5018, 7, 0.85),   # cold Causes freeze
        (8001, 3301, 7, 0.80),   # sun Causes bright
        (8009, 6003, 7, 0.75),   # rain Causes water

        # ── Desires (18) ─────────────────────────────────────────────────
        (1001, 5009, 18, 0.85),  # dog Desires play
        (1001, 6007, 18, 0.90),  # dog Desires food
        (1002, 1004, 18, 0.70),  # cat Desires fish
        (1002, 6007, 18, 0.85),  # cat Desires food
        (1002, 1008, 18, 0.75),  # cat Desires mouse
        (1008, 6006, 18, 0.80),  # mouse Desires cheese
        (1007, 6008, 18, 0.85),  # elephant Desires plant
        (1102, 5009, 18, 0.90),  # child Desires play

        # ── LocatedNear (29) ─────────────────────────────────────────────
        (9001, 9002, 29, 0.80),  # grass LocatedNear tree
        (4007, 4010, 29, 0.75),  # lake LocatedNear forest
    ]

    cur.executemany(
        "INSERT INTO relations (source_id, target_id, relation_type_id, weight) "
        "VALUES (?, ?, ?, ?)",
        relations,
    )

    conn.commit()

    concept_count = cur.execute("SELECT COUNT(*) FROM concepts").fetchone()[0]
    relation_count = cur.execute("SELECT COUNT(*) FROM relations").fetchone()[0]

    conn.close()

    print(f"Micro Bible built: {concept_count} concepts, {relation_count} relations at {db_path}")
