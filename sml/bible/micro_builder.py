"""Micro-PoC Bible builder with ~50 hand-crafted concepts."""

import sqlite3
from sml.bible.schema import create_bible_db


def build_micro_bible(db_path: str) -> None:
    """Build a micro Bible with ~50 concepts and relations for PoC testing."""
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

        # Sizes (domain=4 property, category=2 size)
        (3101, "/c/en/big", "big", "big_3101", 4, 2, 0, 0, "Of considerable size or extent"),
        (3102, "/c/en/small", "small", "small_3102", 4, 2, 0, 0, "Of limited size or extent"),

        # Places (domain=1, category=4 place)
        (4001, "/c/en/park", "park", "park_4001", 1, 4, 0, 0, "A public area of land for recreation"),
        (4002, "/c/en/kitchen", "kitchen", "kitchen_4002", 1, 4, 0, 0, "A room where food is prepared"),
        (4003, "/c/en/school", "school", "school_4003", 1, 4, 0, 0, "An institution for educating children"),
        (4004, "/c/en/ocean", "ocean", "ocean_4004", 1, 4, 0, 0, "A vast body of salt water"),

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

        # Properties - temperature (domain=4, category=3)
        (3201, "/c/en/hot", "hot", "hot_3201", 4, 3, 0, 0, "Having a high temperature"),
        (3202, "/c/en/cold", "cold", "cold_3202", 4, 3, 0, 0, "Having a low temperature"),

        # Properties - speed (domain=4, category=4)
        (3203, "/c/en/fast", "fast", "fast_3203", 4, 4, 0, 0, "Moving at high speed"),
        (3204, "/c/en/slow", "slow", "slow_3204", 4, 4, 0, 0, "Moving at low speed"),

        # Properties - age (domain=4, category=5)
        (3205, "/c/en/old", "old", "old_3205", 4, 5, 0, 0, "Having lived for many years"),
        (3206, "/c/en/new", "new", "new_3206", 4, 5, 0, 0, "Recently made or discovered"),

        # Properties - luminosity (domain=4, category=6)
        (3301, "/c/en/bright", "bright", "bright_3301", 4, 6, 0, 0, "Giving out or reflecting much light"),
        (3302, "/c/en/dark", "dark", "dark_3302", 4, 6, 0, 0, "With little or no light"),

        # Properties - weight (domain=4, category=7)
        (3401, "/c/en/heavy", "heavy", "heavy_3401", 4, 7, 0, 0, "Of great weight; difficult to lift"),
        (3402, "/c/en/light", "light", "light_3402", 4, 7, 0, 0, "Of little weight; not heavy"),

        # Food / Substances (domain=1, category=3 substance)
        (6001, "/c/en/apple", "apple", "apple_6001", 1, 3, 0, 0, "A round fruit with red or green skin"),
        (6002, "/c/en/bread", "bread", "bread_6002", 1, 3, 0, 0, "Food made from flour, water, and yeast"),
        (6003, "/c/en/water", "water", "water_6003", 1, 3, 0, 0, "A transparent liquid essential for life"),
        (6004, "/c/en/milk", "milk", "milk_6004", 1, 3, 0, 0, "White liquid produced by mammals"),
        (6005, "/c/en/ice", "ice", "ice_6005", 1, 3, 0, 0, "Frozen water; a solid form of water"),

        # Abstract (domain=2 abstract)
        (7001, "/c/en/love", "love", "love_7001", 2, 0, 0, 0, "An intense feeling of deep affection"),
        (7002, "/c/en/fear", "fear", "fear_7002", 2, 0, 0, 0, "An unpleasant emotion caused by threat"),
        (7003, "/c/en/knowledge", "knowledge", "knowledge_7003", 2, 0, 0, 0, "Facts or information acquired through experience"),

        # Celestial / Natural (domain=1, category=5)
        (8001, "/c/en/sun", "sun", "sun_8001", 1, 5, 0, 0, "The star at the center of the solar system"),
        (8002, "/c/en/sky", "sky", "sky_8002", 1, 5, 0, 0, "The region of the atmosphere seen from earth"),
        (8003, "/c/en/night", "night", "night_8003", 1, 5, 0, 0, "The period of darkness between sunset and sunrise"),

        # Plants / Nature (domain=1, category=1 living, subcategory=3 plant)
        (9001, "/c/en/grass", "grass", "grass_9001", 1, 1, 3, 0, "Green vegetation covering the ground"),
        (9002, "/c/en/tree", "tree", "tree_9002", 1, 1, 3, 0, "A tall perennial plant with a woody trunk"),

        # Natural phenomena (domain=1, category=5)
        (8004, "/c/en/fire", "fire", "fire_8004", 1, 5, 0, 0, "Combustion producing heat, light, and flame"),
        (8005, "/c/en/snow", "snow", "snow_8005", 1, 5, 0, 0, "Frozen atmospheric water vapor falling as white flakes"),
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
        # Animals capabilities (CapableOf = 5)
        (1001, 5006, 5, 0.95),   # dog CapableOf bark
        (1001, 5002, 5, 0.90),   # dog CapableOf run
        (1001, 5005, 5, 0.70),   # dog CapableOf swim
        (1001, 5008, 5, 0.95),   # dog CapableOf sleep
        (1001, 5009, 5, 0.90),   # dog CapableOf play
        (1001, 5010, 5, 0.95),   # dog CapableOf walk
        (1002, 5007, 5, 0.95),   # cat CapableOf purr
        (1002, 5002, 5, 0.85),   # cat CapableOf run
        (1002, 5008, 5, 0.95),   # cat CapableOf sleep
        (1002, 5012, 5, 0.85),   # cat CapableOf climb
        (1003, 5004, 5, 0.95),   # bird CapableOf fly
        (1004, 5005, 5, 0.98),   # fish CapableOf swim
        (1005, 5005, 5, 0.95),   # penguin CapableOf swim
        (1005, 5010, 5, 0.90),   # penguin CapableOf walk
        (1006, 5005, 5, 0.70),   # snake CapableOf swim
        (1007, 5010, 5, 0.95),   # elephant CapableOf walk
        (1007, 5005, 5, 0.60),   # elephant CapableOf swim
        (1008, 5002, 5, 0.80),   # mouse CapableOf run
        (1008, 5012, 5, 0.75),   # mouse CapableOf climb

        # IsA relations (IsA = 1)
        (1001, 1001, 1, 0.99),   # dog IsA animal (self-ref for simplicity)
        (1002, 1002, 1, 0.99),   # cat IsA animal

        # AtLocation relations (AtLocation = 6)
        (1001, 4001, 6, 0.80),   # dog AtLocation park
        (1001, 2004, 6, 0.85),   # dog AtLocation house
        (1002, 2004, 6, 0.90),   # cat AtLocation house
        (1002, 4002, 6, 0.70),   # cat AtLocation kitchen
        (1003, 4001, 6, 0.75),   # bird AtLocation park
        (1004, 6003, 6, 0.95),   # fish AtLocation water
        (1004, 4004, 6, 0.90),   # fish AtLocation ocean
        (1102, 4003, 6, 0.90),   # child AtLocation school
        (2005, 4003, 6, 0.85),   # book AtLocation school
        (2006, 4002, 6, 0.80),   # table AtLocation kitchen
        (2007, 4002, 6, 0.75),   # chair AtLocation kitchen
        (1005, 4004, 6, 0.85),   # penguin AtLocation ocean
        (1007, 4001, 6, 0.50),   # elephant AtLocation park (zoo/safari)

        # HasProperty relations (HasProperty = 4)
        (1001, 3005, 4, 0.70),   # dog HasProperty brown
        (1001, 3203, 4, 0.75),   # dog HasProperty fast
        (1002, 3102, 4, 0.70),   # cat HasProperty small
        (6001, 3001, 4, 0.80),   # apple HasProperty red
        (6001, 3003, 4, 0.70),   # apple HasProperty green
        (8001, 3004, 4, 0.95),   # sun HasProperty yellow
        (8001, 3201, 4, 0.98),   # sun HasProperty hot
        (8001, 3301, 4, 0.95),   # sun HasProperty bright
        (6003, 3202, 4, 0.60),   # water HasProperty cold
        (8002, 3002, 4, 0.95),   # sky HasProperty blue
        (9001, 3003, 4, 0.95),   # grass HasProperty green
        (8004, 3001, 4, 0.90),   # fire HasProperty red
        (8004, 3201, 4, 0.98),   # fire HasProperty hot
        (8004, 3301, 4, 0.90),   # fire HasProperty bright
        (8005, 3006, 4, 0.98),   # snow HasProperty white
        (8005, 3202, 4, 0.95),   # snow HasProperty cold
        (6005, 3202, 4, 0.98),   # ice HasProperty cold
        (6005, 3006, 4, 0.80),   # ice HasProperty white
        (8003, 3302, 4, 0.95),   # night HasProperty dark
        (4004, 3002, 4, 0.85),   # ocean HasProperty blue
        (9002, 3003, 4, 0.80),   # tree HasProperty green
        (9002, 3101, 4, 0.70),   # tree HasProperty big
        (1007, 3101, 4, 0.95),   # elephant HasProperty big
        (1007, 3401, 4, 0.95),   # elephant HasProperty heavy
        (1008, 3102, 4, 0.90),   # mouse HasProperty small
        (1008, 3402, 4, 0.85),   # mouse HasProperty light
        (1005, 3007, 4, 0.70),   # penguin HasProperty black
        (1005, 3006, 4, 0.70),   # penguin HasProperty white
        (6004, 3006, 4, 0.85),   # milk HasProperty white

        # UsedFor relations (UsedFor = 12)
        (2001, 5009, 12, 0.90),  # ball UsedFor play
        (2005, 7003, 12, 0.85),  # book UsedFor knowledge
        (2003, 5002, 12, 0.60),  # car UsedFor travel (approximate)
        (2006, 5003, 12, 0.70),  # table UsedFor eat
        (2007, 5001, 12, 0.90),  # chair UsedFor sit

        # MadeOf (MadeOf = 16)
        (6002, 6003, 16, 0.60),  # bread MadeOf water (partially)
        (6005, 6003, 16, 0.95),  # ice MadeOf water
        (8005, 6003, 16, 0.90),  # snow MadeOf water

        # Causes (Causes = 7)
        (7002, 5002, 7, 0.60),   # fear Causes run
        (8004, 7002, 7, 0.50),   # fire Causes fear
        (8005, 3202, 7, 0.70),   # snow Causes cold

        # Desires (Desires = 18)
        (1001, 5009, 18, 0.85),  # dog Desires play
        (1002, 1004, 18, 0.70),  # cat Desires fish
        (1102, 5009, 18, 0.90),  # child Desires play
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
