class AbstractBaseline:

    def __init__(self):
        self.pareto_metrics = ['cost', 'reachability', 'validity']

    def get_pareto_front(self, cfs):
        pareto_front = []

        for x in cfs:
            non_dom = False
            for y in cfs:
                any = False
                all = True
                for m in self.pareto_metrics:
                    if x.reward_dict[m] < y.reward_dict[m]:
                        any = True
                    if x.reward_dict[m] > y.reward_dict[m]:
                        all = False

                if any and all:
                    non_dom = True
                    break

            if not non_dom:
                pareto_front.append(x)

        return pareto_front