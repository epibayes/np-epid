import pandas as pd
import numpy as np

class CRKPTransmissionSimulator:
    def __init__(self, path, beta):
        self.path = path
        self.fac_tr = None
        self.floor_tr = None
        self.room_tr = None
        self.known = None
        self.beta = beta
        self.T = self.fac_tr.shape[1]

    def load_data(self):
        # traces
        self.fac_tr = pd.read_csv(
            f"{self.path}/2019-12-18_facility_trace.csv",
            index_col=0, header=0,
            names=np.arange(367)
            )
        self.floor_tr  = pd.read_csv(
            f"{self.path}/2019-12-18_floor_trace.csv",
            index_col=0, header=0,
            names=np.arange(367)
            )
        self.room_tr = pd.read_csv(
            f"{self.path}/2019-12-18_room_trace.csv",
            index_col=0, header=0,
            names=np.arange(367)
            )
        # known events
        self.known = self.load_known()

    def load_known(self):
        # specify fixed points of the simulation
        # e.g., whoever tests upon intake
        # or whoever tests for an invasive infection
        intake_data = pd.read_csv(
            f"{self.path}/intake_data.csv", index_col=0
        )
        tests = pd.read_csv(
            f"{self.path}/infections.csv", index_col=0
        )
        # handle multiple tests same-day per patient: positive result overrules negative
        # assumption is that false negatives are relatively common compared to false positives
        # unresolved issue: what if someone gets a negative test after a positive test?
        tests = tests.reset_index().sort_values(["index", "test_time", "Carbapenem_R"])\
            .drop_duplicates(["index", "test_time"], keep="last").set_index("index")
        
        # generate dataframe of known statuses
        n = self.fac_tr.shape[0]
        T = self.fac_tr.shape[1]
        # this might be more like...known events?
        known_events = pd.DataFrame(index=intake_data.index, columns=range(T))
        for t in range(T):
            # statuses upon intake
            for pid, result in intake_data[intake_data["date"] == t]["crkp"].items():
                known_events.loc[pid, t] = result
            # statuses upon test for infection
            for pid, result in tests[tests["test_time"] == t]["Carbapenem_R"].items():
                known_events.loc[pid, t] = result

        return known_events

    def simulate_data(self, N):
        simulations = []
        # TODO: move generating beta to here?
        for _ in range(N):
            # can try to calculate summary statistics, here
            simulations.append(self.simulate())

    def _simulate(self):
        transmission_status = pd.DataFrame(index=self.known.index, columns=range(self.T))
        # TODO: do i want to hard code week 0?
        transmission_status[0] = self.known[0]
        y = 1/0
        for t in range(1, T):
            status = transmission_status[t-1]
            facility = self.fac_tr[t-1]
            admitted = set(facility[facility!= 0])
            floor = self.floor_tr[t-1]
            room = self.room_tr[t-1]
            # requirement: you were susceptible in the previous timestep 
            # and are not discharged this week
            susceptible = set(status[status==0].index) & admitted
            for j in susceptible:
                hazard_j = 0
                # should not be facility..
                hazard_j += status[status == 1].count() * self.beta[0]
                floor_num = floor.loc[j]
                hazard_j += status[(status == 1) & (floor == floor_num)].count() \
                      * self.beta[floor_num]
                room_num = room.loc[j]
                hazard_j += status[(status == 1) & (room == room_num)].count() \
                    * self.beta[-1]

                p_j = 1 - np.exp(hazard_j)

                Y_j = np.random.binomial(1, p_j)
                transmission_status.loc[j, t] = Y_j

            # TODO: overwrite this with...known statuses?