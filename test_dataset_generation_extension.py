from dataset_generation_extension import *


def build_test_set(turn, test_size, T21ST):

    seed = 721345631

    colors = ["pink", "white", "black", "darkmagenta", "lightblue"]
    cities = ["cosenza", "delhi", "cairo", "mumbai", "moscow", "singapore", "chicago", "toronto", "barcelona"]
    labels = ["wall", "chair", "roof", "flower", "butterfly", "laptop", "desk", "cloud", "storm"]
    attributes = ["surname", "owner", "lake", "hair", "weight", "strength", "quality"]

    predicates = colors + cities + labels + attributes
    closures = ["loops", "family", "trains", "journey"]

    test_tuples = []

    match turn:
        case "core":
            for i in range(test_size):
                np.random.seed(seed % (i + 1) * 19)

                chosen = 0

                questions_assignments, answers_assignments, facts_assignments = label_assignment(predicates,
                                                                                                 np.random.choice(
                                                                                                     predicates), False,
                                                                                                 False)

                test_tuples.append(
                    [questions_assignments[chosen], answers_assignments[chosen], facts_assignments[chosen]])

                questions_prevents, answers_prevents, facts_prevents = prevent_value(predicates,
                                                                                     np.random.choice(predicates),
                                                                                     False, False)

                test_tuples.append([questions_prevents[chosen], answers_prevents[chosen], facts_prevents[chosen]])

                p_1, p_2 = np.random.choice(predicates, 2, replace=False)
                questions_combinations, answers_combinations, facts_combinations = generate_combinations(p_1, p_2,
                                                                                                         False, False)

                test_tuples.append(
                    [questions_combinations[chosen], answers_combinations[chosen], facts_combinations[chosen]])

                p_1, p_2 = np.random.choice(predicates, 2, replace=False)
                questions_join, answers_join, facts_join = execute_join(p_1, p_2, attributes, False, False)

                test_tuples.append([questions_join[chosen], answers_join[chosen], facts_join[chosen]])

                questions_closure, answers_closure, facts_closure = transitive_closure(np.random.choice(closures),
                                                                                       np.random.choice(predicates),
                                                                                       False, False)

                test_tuples.append([questions_closure[chosen], answers_closure[chosen], facts_closure[chosen]])

                questions_preferences, answers_preferences, facts_preferences = preferences(
                    np.random.choice(predicates), predicates, False, False)

                test_tuples.append(
                    [questions_preferences[chosen], answers_preferences[chosen], facts_preferences[chosen]])

                questions_filtering, answers_filtering, facts_filtering = select_value(np.random.choice(predicates),
                                                                                       np.random.choice(predicates),
                                                                                       False, False)

                test_tuples.append([questions_filtering[chosen], answers_filtering[chosen], facts_filtering[chosen]])

                questions_negative_filtering, answers_negative_filtering, facts_negative_filtering = select_by_negative_condition(
                    np.random.choice(predicates), np.random.choice(predicates), predicates, False, False)

                test_tuples.append([questions_negative_filtering[chosen], answers_negative_filtering[chosen],
                                    facts_negative_filtering[chosen]])

                questions_numeric_filtering, answers_numeric_filtering, facts_numeric_filtering = select_by_numeric_condition(
                    np.random.choice(predicates), False, False)

                test_tuples.append([questions_numeric_filtering[chosen], answers_numeric_filtering[chosen],
                                    facts_numeric_filtering[chosen]])

        case "core-invariance":
            prompt_invariance = True

            for i in range(test_size):
                np.random.seed(seed % (i + 1) * 19)

                questions_assignments, answers_assignments, facts_assignments = label_assignment(predicates,
                                                                                                 np.random.choice(
                                                                                                     predicates),
                                                                                                 prompt_invariance,
                                                                                                 T21ST)

                chosen = np.random.randint(0, 21)
                test_tuples.append([questions_assignments[chosen], answers_assignments[chosen], facts_assignments[0]])

                questions_prevents, answers_prevents, facts_prevents = prevent_value(predicates,
                                                                                     np.random.choice(predicates),
                                                                                     prompt_invariance, T21ST)

                chosen = np.random.randint(0, 21)
                test_tuples.append([questions_prevents[chosen], answers_prevents[chosen], facts_prevents[0]])

                p_1, p_2 = np.random.choice(predicates, 2, replace=False)
                questions_combinations, answers_combinations, facts_combinations = generate_combinations(p_1, p_2,
                                                                                                         prompt_invariance,
                                                                                                         T21ST)

                chosen = np.random.randint(0, 21)
                test_tuples.append(
                    [questions_combinations[chosen], answers_combinations[chosen], facts_combinations[0]])

                p_1, p_2 = np.random.choice(predicates, 2, replace=False)
                questions_join, answers_join, facts_join = execute_join(p_1, p_2, attributes, prompt_invariance, T21ST)

                chosen = np.random.randint(0, 21 * 9)

                test_tuples.append([questions_join[chosen], answers_join[chosen], facts_join[chosen]])

                questions_closure, answers_closure, facts_closure = transitive_closure(np.random.choice(closures),
                                                                                       np.random.choice(predicates),
                                                                                       prompt_invariance, T21ST)

                chosen = np.random.randint(0, 21)
                test_tuples.append([questions_closure[chosen], answers_closure[chosen], facts_closure[0]])

                questions_preferences, answers_preferences, facts_preferences = preferences(
                    np.random.choice(predicates), predicates, prompt_invariance, T21ST)

                chosen = np.random.randint(0, 21)
                test_tuples.append([questions_preferences[chosen], answers_preferences[chosen], facts_preferences[0]])

                questions_filtering, answers_filtering, facts_filtering = select_value(np.random.choice(predicates),
                                                                                       np.random.choice(predicates),
                                                                                       prompt_invariance, T21ST)

                chosen = np.random.randint(0, 21)
                test_tuples.append([questions_filtering[chosen], answers_filtering[chosen], facts_filtering[0]])

                questions_negative_filtering, answers_negative_filtering, facts_negative_filtering = select_by_negative_condition(
                    np.random.choice(predicates), np.random.choice(predicates), predicates, prompt_invariance, T21ST)

                chosen = np.random.randint(0, 21)
                test_tuples.append([questions_negative_filtering[chosen], answers_negative_filtering[chosen],
                                    facts_negative_filtering[0]])

                questions_numeric_filtering, answers_numeric_filtering, facts_numeric_filtering = select_by_numeric_condition(
                    np.random.choice(predicates), prompt_invariance, T21ST)

                chosen = np.random.randint(0, 21)
                test_tuples.append([questions_numeric_filtering[chosen], answers_numeric_filtering[chosen],
                                    facts_numeric_filtering[0]])

        case "complex":
            for i in range(test_size):
                np.random.seed(seed % (i + 1) * 19)

                chosen = 0

                p_1, p_2 = np.random.choice(predicates, 2, replace=False)
                questions_jneg, answers_jneg, facts_jneg = join_filtering(p_1, p_2, attributes, predicates)

                test_tuples.append([questions_jneg[chosen], answers_jneg[chosen], facts_jneg[chosen]])

                questions_gc, answers_gc, facts_gc = guessing_constraint(labels, np.random.choice(predicates))

                test_tuples.append([questions_gc[chosen], answers_gc[chosen], facts_gc[chosen]])

                p_1, p_2, p_3 = np.random.choice(predicates, 3, replace=False)
                questions_cnf, answers_cnf, facts_cnf = combination_negative_filtering(labels, p_1, p_2, p_3)

                test_tuples.append([questions_cnf[chosen], answers_cnf[chosen], facts_cnf[chosen]])

    return test_tuples

def main():
    turn = "core"  # "core-invariance" # "complex"

    data_folder = "data"
    if not os.path.exists(data_folder):
        os.makedirs(data_folder)

    test_size = 1000

    match turn:
        case "core":

            test_set_file_name = "test_core.csv"

        case "core-invariance":

            test_set_file_name = "test_core_invariance.csv"

        case "complex":
            test_set_file_name = "test_basecomplex.csv"

        case _:
            print("Selected Turn Not Available")
            sys.exit(1)

    test_tuples = build_test_set(turn, test_size, True)

    print("TEST SET SIZE: ", len(test_tuples))

    test_df = pd.DataFrame(test_tuples, columns=["prompt", "answer", "fact"])

    test_df.to_csv(os.path.join(data_folder, test_set_file_name), index=False)



if __name__ == '__main__':
    main()