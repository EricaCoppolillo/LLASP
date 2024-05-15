

def build_test_set():
    seed = 721345631
    n_size = 1000

    colors = ["pink", "white", "black", "darkmagenta", "lightblue"]
    cities = ["cosenza", "delhi", "cairo", "mumbai", "moscow", "singapore", "chicago", "toronto", "barcelona"]
    labels = ["wall", "chair", "roof", "flower", "butterfly", "laptop", "desk", "cloud", "storm"]
    attributes = ["surname", "owner", "lake", "hair", "weight", "strength", "quality"]

    predicates = colors + cities + labels + attributes
    closures = ["loops", "family", "trains", "journey"]

    test_tuples = []

    # PROBLEMS ORDER IS IMPORTANT

    for i in range(n_size):
        np.random.seed(seed % (i + 1) * 19)

        questions_assignments, answers_assignments, facts_assignments = label_assignment(predicates,
                                                                                         np.random.choice(predicates))

        test_tuples.append([questions_assignments[0], answers_assignments[0], facts_assignments[0]])

        questions_prevents, answers_prevents, facts_prevents = prevent_value(predicates, np.random.choice(predicates))

        test_tuples.append([questions_prevents[0], answers_prevents[0], facts_prevents[0]])

        p_1, p_2 = np.random.choice(predicates, 2, replace=False)
        questions_combinations, answers_combinations, facts_combinations = generate_combinations(p_1, p_2)

        test_tuples.append([questions_combinations[0], answers_combinations[0], facts_combinations[0]])

        p_1, p_2 = np.random.choice(predicates, 2, replace=False)
        questions_join, answers_join, facts_join = execute_join(p_1, p_2, attributes)

        test_tuples.append([questions_join[0], answers_join[0], facts_join[0]])

        questions_closure, answers_closure, facts_closure = transitive_closure(np.random.choice(closures),
                                                                               np.random.choice(predicates))

        test_tuples.append([questions_closure[0], answers_closure[0], facts_closure[0]])

        questions_preferences, answers_preferences, facts_preferences = preferences(np.random.choice(predicates),
                                                                                    predicates)

        test_tuples.append([questions_preferences[0], answers_preferences[0], facts_preferences[0]])

        questions_filtering, answers_filtering, facts_filtering = select_value(np.random.choice(predicates),
                                                                               np.random.choice(predicates))

        test_tuples.append([questions_filtering[0], answers_filtering[0], facts_filtering[0]])

        questions_negative_filtering, answers_negative_filtering, facts_negative_filtering = select_by_negative_condition(
            np.random.choice(predicates), np.random.choice(predicates), predicates)

        test_tuples.append(
            [questions_negative_filtering[0], answers_negative_filtering[0], facts_negative_filtering[0]])

        questions_numeric_filtering, answers_numeric_filtering, facts_numeric_filtering = select_by_numeric_condition(
            np.random.choice(predicates))

        test_tuples.append([questions_numeric_filtering[0], answers_numeric_filtering[0], facts_numeric_filtering[0]])

    return test_tuples


test_tuples = build_test_set()

test_df = pd.DataFrame(test_tuples, columns=["prompt", "answer", "fact"])

# CHANGE PATH IF NEEDED
test_df.to_csv("test_df.csv", index=False)
