PRAGMA foreign_keys = ON;

CREATE TABLE transfers (
    player_id INTEGER,
    transfer_date TEXT,
    transfer_season TEXT,
    from_club_id INTEGER,
    to_club_id INTEGER,
    from_club_name TEXT,
    to_club_name TEXT,
    transfer_fee TEXT,
    market_value_in_eur REAL,
    player_name TEXT,
    FOREIGN KEY (player_id) REFERENCES players(player_id),
    FOREIGN KEY (from_club_id) REFERENCES clubs(club_id),
    FOREIGN KEY (to_club_id) REFERENCES clubs(club_id)
);

CREATE TABLE player_valuations (
    player_id INTEGER,
    date TEXT,
    market_value_in_eur REAL,
    current_club_id INTEGER,
    player_club_domestic_competition_id TEXT,
    FOREIGN KEY (player_id) REFERENCES players(player_id),
    FOREIGN KEY (current_club_id) REFERENCES clubs(club_id),
    FOREIGN KEY (player_club_domestic_competition_id) REFERENCES competitions(competition_id)
);

CREATE TABLE game_lineups (
    game_lineups_id TEXT PRIMARY KEY,
    date TEXT,
    game_id INTEGER,
    player_id INTEGER,
    club_id INTEGER,
    player_name TEXT,
    type TEXT,
    position TEXT,
    number INTEGER,
    team_captain INTEGER,
    FOREIGN KEY (game_id) REFERENCES games(game_id),
    FOREIGN KEY (player_id) REFERENCES players(player_id),
    FOREIGN KEY (club_id) REFERENCES clubs(club_id)
);

CREATE TABLE game_events (
    game_event_id TEXT PRIMARY KEY,
    date TEXT,
    game_id INTEGER,
    minute INTEGER,
    type TEXT,
    club_id INTEGER,
    player_id INTEGER,
    description TEXT,
    player_in_id INTEGER,
    player_assist_id INTEGER,
    FOREIGN KEY (game_id) REFERENCES games(game_id),
    FOREIGN KEY (club_id) REFERENCES clubs(club_id),
    FOREIGN KEY (player_id) REFERENCES players(player_id),
    FOREIGN KEY (player_in_id) REFERENCES players(player_id),
    FOREIGN KEY (player_assist_id) REFERENCES players(player_id)
);

CREATE TABLE club_games (
    game_id INTEGER,
    club_id INTEGER,
    own_goals INTEGER,
    own_position TEXT,
    own_manager_name TEXT,
    opponent_id INTEGER,
    opponent_goals INTEGER,
    opponent_position TEXT,
    opponent_manager_name TEXT,
    hosting TEXT,
    is_win INTEGER,
    FOREIGN KEY (game_id) REFERENCES games(game_id),
    FOREIGN KEY (club_id) REFERENCES clubs(club_id),
    FOREIGN KEY (opponent_id) REFERENCES clubs(club_id)
);

CREATE TABLE appearances (
    appearance_id TEXT PRIMARY KEY,
    game_id INTEGER,
    player_id INTEGER,
    player_club_id INTEGER,
    player_current_club_id INTEGER,
    date TEXT,
    player_name TEXT,
    competition_id TEXT,
    yellow_cards INTEGER,
    red_cards INTEGER,
    goals INTEGER,
    assists INTEGER,
    minutes_played INTEGER,
    FOREIGN KEY (game_id) REFERENCES games(game_id),
    FOREIGN KEY (player_id) REFERENCES players(player_id),
    FOREIGN KEY (player_club_id) REFERENCES clubs(club_id),
    FOREIGN KEY (player_current_club_id) REFERENCES clubs(club_id),
    FOREIGN KEY (competition_id) REFERENCES competitions(competition_id)
);

CREATE TABLE games (
    game_id INTEGER PRIMARY KEY,
    competition_id TEXT,
    season INTEGER,
    round TEXT,
    date TEXT,
    home_club_id INTEGER,
    away_club_id INTEGER,
    home_club_goals INTEGER,
    away_club_goals INTEGER,
    home_club_position INTEGER,
    away_club_position INTEGER,
    home_club_manager_name TEXT,
    away_club_manager_name TEXT,
    stadium TEXT,
    attendance INTEGER,
    referee TEXT,
    url TEXT,
    home_club_formation TEXT,
    away_club_formation TEXT,
    home_club_name TEXT,
    away_club_name TEXT,
    aggregate TEXT,
    competition_type TEXT,
    FOREIGN KEY (competition_id) REFERENCES competitions(competition_id),
    FOREIGN KEY (home_club_id) REFERENCES clubs(club_id),
    FOREIGN KEY (away_club_id) REFERENCES clubs(club_id)
);

CREATE TABLE players (
    player_id INTEGER PRIMARY KEY,
    first_name TEXT,
    last_name TEXT,
    name TEXT,
    last_season INTEGER,
    current_club_id INTEGER,
    player_code TEXT,
    country_of_birth TEXT,
    city_of_birth TEXT,
    country_of_citizenship TEXT,
    date_of_birth TEXT,
    sub_position TEXT,
    position TEXT,
    foot TEXT,
    height_in_cm INTEGER,
    contract_expiration_date TEXT,
    agent_name TEXT,
    image_url TEXT,
    url TEXT,
    current_club_domestic_competition_id TEXT,
    current_club_name TEXT,
    market_value_in_eur REAL,
    highest_market_value_in_eur REAL,
    FOREIGN KEY (current_club_id) REFERENCES clubs(club_id),
    FOREIGN KEY (current_club_domestic_competition_id) REFERENCES competitions(competition_id)
);

CREATE TABLE clubs (
    club_id INTEGER PRIMARY KEY,
    club_code TEXT,
    name TEXT,
    domestic_competition_id TEXT,
    total_market_value TEXT,
    squad_size INTEGER,
    average_age REAL,
    foreigners_number INTEGER,
    foreigners_percentage REAL,
    national_team_players INTEGER,
    stadium_name TEXT,
    stadium_seats INTEGER,
    net_transfer_record TEXT,
    coach_name TEXT,
    last_season INTEGER,
    filename TEXT,
    url TEXT,
    FOREIGN KEY (domestic_competition_id) REFERENCES competitions(competition_id)
);

CREATE TABLE competitions (
    competition_id TEXT PRIMARY KEY,
    competition_code TEXT,
    name TEXT,
    sub_type TEXT,
    type TEXT,
    country_id INTEGER,
    country_name TEXT,
    domestic_league_code TEXT,
    confederation TEXT,
    url TEXT,
    is_major_national_league TEXT
);