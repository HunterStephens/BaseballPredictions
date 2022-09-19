import pybaseball
import pandas as pd
import numpy as np
from scipy.optimize import minimize


def log_liklihood(theta, ws, ls):
    ws_rnks = theta[ws][0]
    ls_rnks = theta[ls][0]
    prob = np.power(1 + np.exp(-(ws_rnks - ls_rnks)), -1)
    log_prob = np.log(prob)
    return -np.sum(log_prob)


def win_probability(r1, r2):
    return 1 / (1 + np.exp(-(r1 - r2)))


class Team:
    def __init__(self, id_in, indx_in, league_in, div_in):
        self.id = id_in
        self.index = indx_in
        self.rating = 0.0
        self.schedule = None
        self.wins = 0
        self.losses = 0
        self.league = league_in
        self.division = div_in


class Division:
    def __init__(self, league_name, region_name):
        self.league = league_name
        self.region = region_name
        self.teams = []


class League:
    def __init__(self, year=2022):
        # --- load team id's ---
        data = pd.read_csv("./data/team_ids.csv")
        self.team_ids = data["Team"]
        league_tags = data["League"]
        div_tags = data["Division"]
        self.teams = []
        for i, id in enumerate(self.team_ids):
            self.teams.append(Team(id, i, league_tags[i], div_tags[i]))

        # --- fill schedule ---
        self.year = year
        self.fill_schedule(self.year)

        # --- create divisions ---
        self.divisions = [Division("AL", "East"), Division("AL", "Central"), Division("AL", "West")]
        self.divisions.append([Division("NL", "East"), Division("NL", "Central"), Division("NL", "West")])

    def fill_schedule(self, year):
        # --- go through each team to get schedule and record ---
        for team in self.teams:
            data = pybaseball.schedule_and_record(year, team.id)
            team.schedule = data

    def get_teams(self):
        return self.teams

    def calc_ratings(self):
        bpi = BPI()
        bpi.set_league(mlb)
        ratings = bpi.run()
        for i, team in enumerate(self.teams):
            team.rating = ratings[i]

    def get_rankings(self):
        ids = []
        ratings = []
        for team in self.teams:
            ids.append(team.id)
            ratings.append(team.rating)

        return sorted(zip(ratings, ids), reverse=True)

    def simulate_season(self):
        for n in range(100):
            for team in self.teams:
                for index, gm in team.schedule.iterrows():
                    win = gm["W/L"]
                    if win is not None:
                        if "W" in win:
                            team.wins = team.wins + 1 / 100
                        else:
                            team.losses = team.losses + 1 / 100
                    else:
                        # --- simulate ---
                        opp_tm = gm["Opp"]
                        opp_index = self.team_ids.tolist().index(opp_tm)
                        opp_rtng = self.teams[opp_index].rating
                        win_prob = win_probability(team.rating, opp_rtng)
                        dice = np.random.rand()
                        if dice <= win_prob:
                            team.wins = team.wins + 1 / 100
                        else:
                            team.losses = team.losses + 1 / 100

        ids = []
        wins = []
        losses = []
        for team in self.teams:
            ids.append(team.id)
            wins.append(np.round(team.wins, 2))
            losses.append(np.round(team.losses, 2))

        return sorted(zip(wins, losses, ids), reverse=True)


class BPI:
    def __init__(self):
        self.league = None

    def set_league(self, league_in):
        self.league = league_in

    def run(self):
        # --- initialize_game_arrays ---
        [theta, ws, ls] = self.initialize_game_arrays()

        # --- minimize ---
        res = minimize(log_liklihood, theta, args=(ws, ls))
        return res.x

    def initialize_game_arrays(self):
        # --- get teams ---
        teams = self.league.teams
        ids = self.league.team_ids.tolist()

        # --- initialize vectors ---
        theta_not = np.zeros(len(teams))
        winners = []
        losers = []
        for i in range(len(theta_not)):
            theta_not[i] = teams[i].rating

        for i in range(len(teams)):
            tm_i = teams[i]
            for index, gm in tm_i.schedule.iterrows():
                base_tm = gm["Tm"]
                opp_tm = gm["Opp"]
                win = gm["W/L"]
                if win is not None:
                    # --- make sure opponents schedule has not been added ---
                    opp_index = ids.index(opp_tm)
                    base_index = i
                    if opp_index > i:
                        if "W" in win:
                            winners.append(base_index)
                            losers.append(opp_index)
                        else:
                            losers.append(base_index)
                            winners.append(opp_index)
        return [theta_not, np.array([winners]), np.array([losers])]


if __name__ == "__main__":
    mlb = League(year=2022)
    mlb.calc_ratings()
    rankings = mlb.get_rankings()
    for i in range(len(rankings)):
        print(f"{rankings[i][1]}: {rankings[i][0]}")

    standings = mlb.simulate_season()
    for i in range(len(standings)):
        print(f"{standings[i][2]}: {standings[i][0]}-{standings[i][1]}")
