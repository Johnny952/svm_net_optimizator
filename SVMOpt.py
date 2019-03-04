import numpy as np
import random
import SVMNet
from SVMNet import extract_classes
from sklearn import svm
from sklearn.metrics import precision_score


class SVMOpt(object):
    def __init__(self):
        self.classes = []
        self.selection = 'tournament'

    def set_selection(self, select):
        """
        Set selection attribute.
        :param select: string selection, it can be 'tournament', 'roulette' or 'rank'.
        """
        self.selection = select

    def Mutation(self, svm_net, pmut):
        """
        Mutates a net.
        :param svm_net: The net to mutate.
        :param pmut: The probability of mutation.
        :return: the old net or a new one depending on the Pmut.
        """
        if random.random() <= pmut:
            classes = np.concatenate([svm_net.left_classes, svm_net.right_classes], axis=None)
            aux = SVMNet.SVMNet().new_net(classes)
            return aux
        return svm_net

    def Crossing(self, net1, net2, pcros):
        """
        If the random if less than Pcros, coss the two parents, otherwise return one of the parents.
        :param net1: SVMNet Parent 1.
        :param net2: SVMNet Parent 2.
        :param pcros: The probability of crossing.
        :return: SVMNet Son.
        """
        if random.random() > pcros:
            return random.choice([net1, net2])
        return self.cross(net1, net2, SVMNet.SVMNet())

    def cross(self, net1, net2, son):
        """
        Cross the parents.
        :param net1: SVMNet Parent 1.
        :param net2: SVMNetParent 2.
        :param son: SVMNet Son.
        :return: SVMNet Son
        """
        son.C = (net1.C + net2.C) / 2.0
        son.kernel = random.choice([net1.kernel, net2.kernel])
        son.degree = np.floor((net1.degree + net2.degree) / 2).astype(np.int)
        # son.gamma = (net1.gamma + net2.gamma)/2.0
        son.SVM = svm.SVC(kernel=son.kernel, C=son.C, degree=son.degree, gamma='auto')
        son.left_classes = net1.left_classes
        son.right_classes = net1.right_classes
        if net1.left is not None and net2.left is not None:
            son.left = SVMNet.SVMNet()
            self.cross(net1.left, net2.left, son.left)
        if net1.right is not None and net2.right is not None:
            son.right = SVMNet.SVMNet()
            self.cross(net1.right, net2.right, son.right)
        self.refill(son)
        return son

    def refill(self, net):
        """
        If the node has more than one class and it does not have left and right nodes, it creates them ramdomly.
        :param net: a SVMNet.
        """
        if net.left is None and net.left_classes.size > 1:
            net.left = SVMNet.SVMNet().new_net(net.left_classes)
        elif net.left is not None and net.left_classes.size > 1:
            self.refill(net.left)
        if net.right is None and net.right_classes.size > 1:
            net.right = SVMNet.SVMNet().new_net(net.right_classes)
        elif net.right is not None and net.right_classes.size > 1:
            self.refill(net.right)

    def fitness(self, net, train_set, train_labels, validation_set, validation_labels):
        """
        Computes the accuracy of a net.
        :param net: SVMNet to Compute its accuracy.
        :param train_set: Training examples.
        :param train_labels: Training labels of each example in train_set.
        :param validation_set: Validation examples.
        :param validation_labels: Validation labels of each example in validation_set.
        :return: the accuracy of the net.
        """
        net.fit(train_set, train_labels)
        pred = net.classify(validation_set)
        # Changeable average to 'macro', 'micro' or 'weighted'
        return precision_score(validation_labels, pred, average='micro')

    def tournament_selection(self, population, pcros, train_set, train_labels, validation_set, validation_labels, k):
        """
        Tournament selection of k nets.
        :param population: List of nets.
        :param pcros: Probability of crossing.
        :param train_set: Training examples.
        :param train_labels: Training examples labels.
        :param validation_set: Validation examples.
        :param validation_labels: Validation examples labels.
        :param k: Numbers of nets to be selected.
        :return: List of new nets of the next generation.
        """
        quantity_nets = len(population)
        new_population = []
        while len(new_population) < quantity_nets:
            # Random selection of k nets
            selected = []
            k_fitness = []
            nList = np.arange(quantity_nets)
            for i in range(k):
                p = random.choice(nList)
                nList = np.setdiff1d(nList, p)
                # nList.remove(p)
                selected.append(population[p])
                f = self.fitness(population[p], train_set, train_labels, validation_set, validation_labels)
                k_fitness.append(f)
            # Sort nets by its fitness
            index = np.flip(np.argsort(k_fitness), axis=0)
            sorted_selected = []
            for i in index:
                sorted_selected.append(selected[i])
            best_selected = sorted_selected[0]
            new_population.append(best_selected)
            sorted_selected = sorted_selected[1:]
            # Cross every net in old population with the best net
            for s in sorted_selected:
                son = self.Crossing(best_selected, s, pcros)
                new_population.append(son)
                if len(new_population) >= k:
                    break
        return new_population

    def roulette_wheel_selection(self, population, pcros, train_set, train_labels, validation_set, validation_labels):
        """
        Roulette wheel Selection.
        :param population: List of nets.
        :param pcros: Probability of crossing.
        :param train_set: Training examples.
        :param train_labels: Training labels.
        :param validation_set: Validation examples.
        :param validation_labels: Validation labels.
        :return: The next generation population
        """
        quantity_nets = len(population)
        new_population = []
        n_fitness = []
        # Computes the fitness for each net in population
        for p in population:
            f = self.fitness(p, train_set, train_labels, validation_set, validation_labels)
            n_fitness.append(f)
        # Sort by fitness
        index = np.flip(np.argsort(n_fitness), axis=0)
        sorted_population = []
        probabilities = []
        total = np.sum(n_fitness)
        # Computes the probability to be selected for each net in population depending on the fitness
        for i in index:
            sorted_population.append(population[i])
            probabilities.append(1.0 * n_fitness[i] / total)
        cummulative_sum = np.cumsum(probabilities)
        while len(new_population) < quantity_nets:
            # Spin the roulette
            rand = random.random() * (cummulative_sum[0] - cummulative_sum[-1]) + cummulative_sum[-1]
            selected = sorted_population[0]
            for c in range(quantity_nets - 1):
                if cummulative_sum[c] > rand and cummulative_sum[c + 1] <= rand:
                    selected = sorted_population[c]
                    break
            rand_index = random.randint(0, quantity_nets - 1)
            new_population.append(self.Crossing(selected, population[rand_index], pcros))
        return new_population

    def exponential_rank_selection(self, population, pcros, train_set, train_labels, validation_set, validation_labels):
        """
        Exponential Rank Selection.
        :param population: List of nets.
        :param pcros: Probability of crossing
        :param train_set: Training examples.
        :param train_labels: Training labels.
        :param validation_set: Validation examples.
        :param validation_labels: Validation labels.
        :return: The next generation population.
        """
        quantity_nets = len(population)
        new_population = []
        n_fitness = []
        # Computes the fitness for each net in population.
        for p in population:
            f = self.fitness(p, train_set, train_labels, validation_set, validation_labels)
            n_fitness.append(f)
        # Sort by fitness
        index = np.flip(np.argsort(n_fitness), axis=0)
        c = (2 * quantity_nets * (quantity_nets - 1)) / (6 * (quantity_nets - 1) + quantity_nets)
        probability = []
        sorted_population = []
        position = 1
        # Computes the probability to be selected depending on the rank
        for i in index:
            sorted_population.append(population[i])
            probability.append(np.exp(-1.0 * position / c))
        while len(new_population) < quantity_nets:
            alpha = random.random() * (2 / c - c / 9) + c / 9
            selected = sorted_population[0]
            for j in range(quantity_nets):
                if probability[j] < alpha:
                    selected = sorted_population[j]
                    break
            rand_index = random.randint(0, quantity_nets - 1)
            new_population.append(self.Crossing(selected, population[rand_index], pcros))
        return new_population

    def run(self, trainSet, trainLabels, validationSet, validationLabels, Npop, Pcrossing=0.8, Pmutation=0.2, Ngen=15, Ntor=20,
            exponential=False, Acros=0.25, Amut=0.25):
        """
        Optimization method
        :param trainSet: Training examples.
        :param trainLabels: Training labels.
        :param validationSet: Validation examples.
        :param validationLabels: Validation labels.
        :param Npop: Number of nets in population.
        :param Pcrossing: Probability of crossing.
        :param Pmutation: Probability if mutation.
        :param Ngen: Number of generations.
        :param Ntor: Number of nets in tournament selection, ignored in others selections.
        :param exponential: False if Pcrossing and Pmutation are constant, True if Pcrossing increase exponentially to 1
        and Pmutation decrease the same way to 0.
        :param Acros: How much increase Pcros in every generation.
        :param Amut: How much decrease Pmut in every generation.
        :return: The best nets got, a list of the mean fitness in every generation, a list of the max fitness
        in every generation
        """
        initial_population = []
        c = extract_classes(trainLabels)
        max_fitness = []
        mean_fitness = []
        # Creates initial population
        for i in range(Npop):
            net = SVMNet.SVMNet().new_net(c)
            initial_population.append(net)
        best_net = initial_population[0]
        best_fitness = 0
        for generation in range(Ngen):
            print("Actual generation: ", generation)
            if exponential:
                Pcros = (1 - Pcrossing) * (1 - np.exp(-Acros * generation)) + Pcrossing
                Pmut = Pmutation * np.exp(-Amut * generation)
            else:
                Pcros = Pcrossing
                Pmut = Pmutation
            # Selection and crossing
            if self.selection == 'tournament':
                new_population = self.tournament_selection(initial_population, Pcros, trainSet, trainLabels,
                                                           validationSet, validationLabels, k=Ntor)
            elif self.selection == 'roulette':
                new_population = self.roulette_wheel_selection(initial_population, Pcros, trainSet, trainLabels,
                                                               validationSet, validationLabels)
            elif self.selection == 'rank':
                new_population = self.exponential_rank_selection(initial_population, Pcros, trainSet, trainLabels,
                                                                 validationSet, validationLabels)
            else:
                raise ValueError("Selection not supported")
            # Mutation
            muted_population = []
            for i in range(Ntor):
                mutedSon = self.Mutation(new_population[i], Pmut)
                muted_population.append(mutedSon)
            # Change old population by the new one
            initial_population = np.copy(muted_population)
            meanFitness = 0
            maxFitness = 0
            best = initial_population[0]
            N = len(initial_population)
            # Computes the max and mean fitness
            for net in initial_population:
                fit = self.fitness(net, trainSet, trainLabels, validationSet, validationLabels)
                meanFitness += fit * 1.0 / N
                if maxFitness <= fit:
                    maxFitness = fit
                    best = net
            print("Max fitness: ", maxFitness)
            mean_fitness.append(meanFitness)
            max_fitness.append(maxFitness)
            if maxFitness >= best_fitness:
                best_fitness = maxFitness
                best_net = best
        print("Best fitness: ", best_fitness)
        return best_net, mean_fitness, max_fitness

    def set_encode(self, labels):
        """
        Set classes attribute.
        :param labels: Training labels.
        """
        self.classes = np.unique(labels)

    def encode_labels(self, labels):
        """
        Encode string labels.
        :param labels: labels to be encoded.
        :return: encoded labels from 0 to n-1, n equal to the number of classes.
        """
        new_labels = []
        for label in labels:
            for c in range(len(self.classes)):
                if label == self.classes[c]:
                    new_labels.append(c)
        return new_labels