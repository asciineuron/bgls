Search.setIndex({"docnames": ["clifford_decomposition", "clifford_simulation", "features", "how_it_works", "index", "mps_simulation", "qaoa_example", "start", "tips", "using_custom_states", "when_to_use"], "filenames": ["clifford_decomposition.ipynb", "clifford_simulation.ipynb", "features.ipynb", "how_it_works.ipynb", "index.rst", "mps_simulation.ipynb", "qaoa_example.ipynb", "start.ipynb", "tips.ipynb", "using_custom_states.ipynb", "when_to_use.ipynb"], "titles": ["Clifford Decomposition", "Using BGLS on Clifford and near-Clifford circuits", "Features", "How <code class=\"docutils literal notranslate\"><span class=\"pre\">BGLS</span></code> works", "Documentation for <cite>BGLS</cite>", "Using BGLS to sample from Matrix Product State Circuits", "Using BGLS with MPS to solve QAOA problems", "Getting started", "Tips for using <code class=\"docutils literal notranslate\"><span class=\"pre\">BGLS</span></code>", "Using custom states with <code class=\"docutils literal notranslate\"><span class=\"pre\">BGLS</span></code>", "When to use <code class=\"docutils literal notranslate\"><span class=\"pre\">BGLS</span></code>"], "terms": {"here": [0, 1, 2, 3, 5, 6, 7, 8, 9], "we": [0, 1, 2, 3, 5, 6, 7, 8, 9, 10], "us": [0, 3, 4], "near_clifford_solv": 0, "circuit_clifford_decomposit": 0, "function": [0, 1, 2, 3, 5, 6, 7, 8, 9, 10], "which": [0, 1, 2, 3, 5, 7, 8], "return": [0, 1, 2, 3, 4, 5, 6, 7, 9], "list": [0, 3, 5, 6, 9], "all": [0, 2, 4, 5, 6, 8, 9], "f": [0, 2, 5, 7, 8, 10], "2": [0, 1, 2, 3, 5, 6, 7, 8, 9, 10], "non": [0, 1, 2], "circuit": [0, 3, 4, 6, 7, 9, 10], "approxim": [0, 1], "input": [0, 7], "fidel": 0, "import": [0, 1, 2, 3, 5, 6, 7, 8, 9], "numpi": [0, 1, 3, 5, 6, 7, 8, 9], "np": [0, 1, 3, 5, 6, 7, 8, 9], "matplotlib": [0, 1, 3, 5, 6, 8], "pyplot": [0, 1, 3, 5, 6, 8], "plt": [0, 1, 3, 5, 6, 8], "cirq": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], "bgl": 0, "test": [0, 1, 5], "first": [0, 2, 3, 5, 6, 8, 9], "just": [0, 1, 2, 5, 7], "defin": [0, 4, 6, 7], "some": [0, 4, 7], "helper": [0, 9], "quantifi": 0, "how": [0, 1, 4, 5, 10], "well": [0, 1, 5, 6, 8], "our": [0, 1, 3, 5, 6, 9], "result": [0, 1, 2, 4, 5, 6, 7, 9], "match": [0, 1, 5, 6], "either": [0, 3], "each": [0, 1, 2, 3, 4, 5, 6], "other": [0, 5, 7, 8], "predict": 0, "distribut": [0, 1, 2, 3, 4, 5, 6, 7, 8, 10], "def": [0, 1, 3, 5, 6, 7, 9], "result_overlap": 0, "resultdict": [0, 6], "b": [0, 1, 2, 3, 6, 7, 8, 9], "fraction": [0, 1], "those": 0, "i": [0, 1, 3, 4, 5, 6, 7, 8, 9, 10], "e": [0, 1, 2, 5, 6, 8, 9], "go": 0, "through": [0, 1, 2, 3, 5, 6], "everi": [0, 5], "see": [0, 1, 2, 3, 5, 7, 8, 10], "": [0, 1, 3, 4, 5, 6, 7, 8, 9, 10], "new": [0, 3], "same": [0, 2, 5, 6, 10], "total_measur": 0, "0": [0, 1, 2, 3, 5, 6, 7, 8, 9, 10], "matching_measur": 0, "used_b_indic": 0, "keep": 0, "track": [0, 1], "so": [0, 2, 9, 10], "doubl": 0, "count": [0, 1, 3], "uniqu": [0, 3], "gate_kei": 0, "measur": [0, 1, 3, 4, 5, 6, 7, 8, 9], "item": [0, 1, 9], "try": [0, 1], "b_re": 0, "except": [0, 1, 5], "print": [0, 1, 2, 3, 6, 7, 9], "kei": [0, 1, 2, 5, 6, 7, 9], "found": [0, 6, 8], "second": [0, 3], "repa": 0, "rang": [0, 1, 2, 3, 5, 6, 7, 8, 9], "shape": [0, 5, 6, 9], "1": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], "repb": 0, "append": [0, 1, 5, 6, 8, 9], "break": 0, "theoretic_overlap": [0, 1], "prob_vec": [0, 1], "ndarrai": [0, 1, 9], "percent": [0, 1], "overlap": [0, 1], "sampl": [0, 1, 2, 4, 6, 7, 8], "exact": [0, 1], "probabl": [0, 1, 2, 3, 5, 6, 7, 8, 9, 10], "from": [0, 1, 2, 3, 4, 6, 7, 8, 9, 10], "state": [0, 1, 3, 4, 6, 7, 8, 10], "vector": [0, 1, 3, 4, 5, 7, 9, 10], "ideal": [0, 1], "num_match": [0, 1], "histogram": [0, 1, 5], "mea": [0, 1], "tot_sampl": [0, 1], "sum": [0, 1, 3, 6, 10], "valu": [0, 1, 6, 8, 9], "bitstr": [0, 1, 2, 3, 5, 6, 7, 8, 9, 10], "cnt": [0, 1], "theoretic_cnt": [0, 1], "int": [0, 1, 3, 4, 5, 6, 7, 8, 9], "min": [0, 1], "exact_overlap": 0, "prob_vec1": 0, "prob_vec2": 0, "give": 0, "correspond": [0, 5], "two": [0, 3, 8], "final": [0, 3, 4, 5, 6, 7, 10], "when": [0, 2, 4, 5, 8, 9], "one": [0, 1, 2, 3, 5, 6, 8, 9], "an": [0, 1, 2, 3, 5, 8, 9, 10], "total_prob_match": 0, "len": [0, 1, 2, 3, 5, 6, 9], "ab": [0, 1, 3, 5, 6, 7, 9], "max": [0, 6, 8], "possibl": [0, 2], "henc": [0, 5], "total": [0, 1, 8], "show": [0, 1, 5, 7, 8, 9], "can": [0, 1, 3, 4, 5, 6, 7, 8, 9, 10], "expand": [0, 1], "origin": 0, "initi": [0, 3, 5, 6, 7, 9], "random": [0, 1, 3, 5, 6, 8], "g": [0, 5, 6, 7, 8, 9], "t": [0, 1, 2, 8, 9, 10], "gate": [0, 1, 4, 5, 6, 7, 8, 9], "qubit": [0, 1, 2, 4, 5, 6, 7, 8, 9], "linequbit": [0, 1, 2, 3, 5, 6, 7, 8, 9], "3": [0, 1, 3, 5, 6, 7, 8], "domain": 0, "h": [0, 1, 2, 3, 5, 6, 7, 8, 9], "cnot": [0, 1, 2, 3, 5, 7, 8, 9], "clifford_circuit": 0, "generate_random_circuit": 0, "n_moment": [0, 1, 5, 8], "10": [0, 1, 2, 5, 6, 7, 8], "op_dens": [0, 1, 5, 8], "5": [0, 1, 5, 6, 8], "gate_domain": [0, 1, 5], "random_st": [0, 1], "gener": [0, 1, 2, 5, 8, 10], "strict": 0, "expans": [0, 1], "term": [0, 1], "expanded_circuit": 0, "expanded_amplitud": 0, "compar": [0, 10], "have": [0, 2, 3, 5, 9, 10], "done": 0, "befor": [0, 8], "cirq_sim": 0, "simul": [0, 1, 3, 5, 6, 8, 9, 10], "cirq_result": 0, "run": [0, 1, 2, 4, 7, 8, 9], "repetit": [0, 1, 2, 4, 5, 6, 7, 8, 9], "1000": [0, 1, 3, 6, 8, 9], "number": [0, 1, 4, 5, 6, 9, 10], "trial": [0, 8], "vec": 0, "power": [0, 1, 5, 6, 7], "final_state_vector": [0, 1, 3], "_": [0, 1, 3, 5, 6, 8, 9], "plot_state_histogram": [0, 1, 2, 5, 7, 9], "subplot": [0, 1, 5], "str": [0, 1, 3, 4, 5, 6, 7, 9], "97825": 0, "0732233": 0, "4267766": 0, "repeat": 0, "randomli": [0, 5], "standard": [0, 5], "per": [0, 1, 2, 5, 6], "usual": [0, 3, 9, 10], "bgls_sim": 0, "stabilizerchformsimulationst": [0, 1], "initial_st": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], "appli": [0, 1, 2, 5, 7, 8, 9, 10], "act_on_near_clifford": [0, 1], "born": [0, 1, 2, 5, 7, 8], "compute_probability_stabilizer_st": [0, 1], "seed": [0, 1, 4, 6], "bgls_result": 0, "646": 0, "contrast": [0, 3], "thi": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], "follow": [0, 1, 2, 3, 5, 7, 8, 9], "ensur": 0, "path": [0, 1, 10], "ar": [0, 1, 2, 3, 7, 8, 9], "cover": 0, "expanded_result": 0, "none": [0, 4, 6, 7, 9], "c": [0, 1], "els": [0, 3, 5, 6], "76575": 0, "note": [0, 2, 5, 6, 7, 8, 9], "mai": [0, 4, 8], "actual": [0, 5], "recreat": [0, 5], "wors": 0, "than": [0, 4, 5, 7, 10], "becaus": [0, 2, 3, 5], "wherea": [0, 5, 10], "latter": 0, "yield": [0, 5, 6], "accord": [0, 2], "rel": [0, 10], "amplitud": [0, 1, 3, 5, 6, 10], "even": [0, 4], "doesn": [0, 1], "explor": [0, 5], "entir": [0, 10], "space": 0, "method": [0, 2, 4, 6, 9], "equal": [0, 2, 5, 8, 9], "captur": 0, "vari": [0, 8], "reincorpor": 0, "weighted_expanded_result": 0, "rng": [0, 1], "randomst": [0, 5, 6], "expanded_prob": 0, "rep": 0, "chosen_circuit": 0, "choic": [0, 3], "p": [0, 2, 3, 6, 7], "634": 0, "know": [0, 7], "optim": [0, 1, 4, 6], "doe": [0, 1, 2], "mean": [0, 5], "alwai": 0, "veri": [0, 5], "good": [0, 5, 7], "pick": [0, 1, 6], "close": 0, "case": [0, 2, 5, 8, 10], "better": [0, 5], "For": [0, 1, 2, 3, 5, 6, 9, 10], "exampl": [0, 2, 5, 7, 8, 9], "rz": 0, "theta": [0, 1], "epsilon": 0, "1e": 0, "8": [0, 5, 6, 8], "op": [0, 1, 3, 5, 6, 9], "isinst": 0, "common_g": 0, "zpowgat": 0, "expon": 0, "25": [0, 8], "rad": 0, "rze": 0, "map_oper": 0, "expanded_rz": 0, "rze_amplitud": 0, "98375": 0, "9845": 0, "now": [0, 2, 3, 5, 9, 10], "readi": 0, "larger": [0, 1], "behavior": [0, 1], "creat": [0, 1, 3, 4, 5, 6, 8, 9], "over": [0, 3, 6, 10], "increas": [0, 1, 5], "both": [0, 1, 5, 10], "statevector": [0, 7, 9, 10], "sampler": [0, 4, 5, 6], "compq": 0, "comp_circuit": 0, "comp_circuit_clifford": 0, "overlapscirq": 0, "overlapsbgl": 0, "overlapscirq_clifford": 0, "overlapsbgls_clifford": 0, "linspac": [0, 6, 8], "start": [0, 1, 3, 4, 5, 6, 8, 9, 10], "stop": [0, 6], "1500": 0, "num": 0, "dtype": [0, 6, 7, 8, 9], "ressim": 0, "re": [0, 6], "resbgl": 0, "ressim_clifford": 0, "vec_clifford": 0, "res_clifford": 0, "resbgls_clifford": 0, "plot": [0, 1, 5, 8], "color": [0, 5, 6, 8], "label": [0, 1, 5, 8], "approx": 0, "r": [0, 1, 6, 9, 10], "y": [0, 5], "legend": [0, 5, 8], "xlabel": [0, 1, 3, 6, 8], "ylabel": [0, 1, 3, 6], "titl": [0, 1, 6], "type": [0, 1, 2, 7, 9], "v": [0, 1, 6], "tend": 0, "toward": 0, "fluctuat": 0, "decreas": [0, 1, 8], "howev": [0, 1, 5], "while": [0, 2], "lag": 0, "price": 0, "pai": 0, "would": [0, 5], "lose": 0, "benefit": 0, "work": [0, 4, 5, 7, 9], "next": 0, "similarli": 0, "sweep": [0, 6], "across": [0, 6], "angl": 0, "again": [0, 5, 8], "4": [0, 1, 5, 6, 8], "rzi": 0, "\u03b8": 0, "clear": 0, "error": 0, "metric": 0, "paper": 0, "investig": 0, "ha": [0, 5], "fulli": [0, 5], "sub": 0, "allow": 0, "speedup": [0, 5, 8], "cost": [0, 4, 6], "output": [0, 1, 9], "accuraci": 0, "cliffq": 0, "cliff_circuit": 0, "20": [0, 1, 5], "cirq_clifford_sim": 0, "cliffordsimul": 0, "cirq_exact_sim": 0, "exact_sim": 0, "exact_vec": 0, "final_state_vec": 0, "zeros_lik": 0, "expanded_cliff": 0, "cliff_amplitud": 0, "final_st": [0, 2], "state_vector": [0, 5], "final_prob_vec": 0, "thu": [0, 1, 5, 10], "retain": 0, "more": [0, 4], "grant": 0, "comput": [0, 1, 2, 3, 5, 6, 7, 8, 9, 10], "time": [0, 1, 2, 3, 4, 5, 6], "what": [0, 7], "gain": [0, 5], "abil": 0, "effici": [0, 1, 6], "pure": [0, 1], "perform": [0, 5, 8], "demonstr": 1, "particular": [1, 5], "univers": 1, "set": [1, 6, 10], "explicitli": 1, "group": 1, "consist": 1, "hadamard": [1, 5, 8], "check": [1, 3, 9], "oper": [1, 2, 3, 4, 5, 7, 8, 9, 10], "satisfi": 1, "criterion": 1, "via": [1, 2, 5, 7, 8], "protocol": [1, 2, 3, 6, 7, 8], "has_stabilizer_effect": 1, "assert": [1, 9], "To": [1, 5], "compos": 1, "take": [1, 5, 6, 8], "advantag": [1, 5, 7, 10], "stabil": 1, "In": [1, 2, 3, 5, 6, 7, 8, 10], "make": [1, 5, 6, 10], "ch": 1, "form": [1, 4, 10], "introduct": 1, "arxiv": [1, 3, 7], "1808": 1, "00128": 1, "addition": 1, "overal": 1, "phase": 1, "enabl": 1, "reconstruct": 1, "alreadi": [1, 9], "implement": [1, 3, 4, 6, 7, 8], "sim": [1, 2, 6, 9, 10], "provid": [1, 4, 7], "cirq_stabilizer_ch_bitstring_prob": 1, "given": [1, 4, 5, 7, 9], "basi": [1, 5], "trivial": 1, "101": [1, 6], "base": [1, 5], "ch_stab_stat": 1, "certain": 1, "look": [1, 5], "much": [1, 5], "substitut": [1, 4], "abov": [1, 2, 3, 5, 7, 8, 9, 10], "notabl": 1, "still": [1, 2], "support": [1, 3, 8, 9, 10], "act_on": [1, 2, 3, 5, 6, 7, 8], "ch_stabilizer_simul": 1, "m": [1, 5, 6, 7], "x": [1, 2, 3, 5, 6, 7, 8, 9, 10], "rigid": 1, "cannot": [1, 5, 10], "handl": 1, "z": [1, 2, 5, 6, 7, 9], "typeerror": 1, "upon": [1, 8], "apply_near_clifford_g": 1, "As": [1, 3, 5, 8, 9, 10], "ani": [1, 4, 5, 7, 9], "singl": [1, 5], "diagon": [1, 6, 8], "rotat": 1, "decompos": 1, "co": 1, "sin": 1, "sqrt": [1, 3, 6], "pi": 1, "obtain": [1, 2, 8], "includ": [1, 2, 7, 10], "equival": [1, 5, 8, 9], "Then": [1, 5], "whenev": 1, "encount": 1, "mere": 1, "weight": 1, "respect": 1, "textrm": 1, "branch": 1, "out": [1, 3, 6, 8], "random_circuit": [1, 5, 8], "With": [1, 7], "strictli": 1, "increasingli": 1, "deep": 1, "runtim": [1, 5, 6, 8], "depth": [1, 8, 10], "scale": [1, 5, 6], "essenti": [1, 5], "linearli": 1, "end": [1, 5], "convers": [1, 5], "fix": [1, 5, 6], "sinc": [1, 3, 6, 8], "onli": [1, 2, 8], "exponenti": [1, 5, 10], "grow": [1, 10], "expect": [1, 2, 3, 5, 6, 10], "requir": [1, 9], "o": [1, 8], "accur": 1, "num_t": 1, "100": [1, 2, 6, 8], "insert": 1, "randint": 1, "low": [1, 5], "high": 1, "batch_insert": 1, "begin": [1, 5], "posit": 1, "affect": 1, "preserv": 1, "exactsim": 1, "exactr": 1, "statevec": [1, 5], "xlim": 1, "xmin": 1, "ylim": 1, "ymin": 1, "moment": 1, "sever": 2, "describ": [2, 3, 5, 8], "below": [2, 3, 5, 8, 9, 10], "setup": [2, 3, 4, 7, 8], "store": [2, 10], "statevectorsimulationst": [2, 3, 5, 7, 8], "compute_prob": [2, 4, 7, 8], "compute_probability_state_vector": [2, 5, 7, 8], "dirac_not": 2, "target_tensor": [2, 3, 7], "flatten": 2, "0000001111": 2, "71": [2, 6], "00": [2, 7, 8], "11": [2, 5, 6], "although": 2, "multipl": [2, 9], "parallel": [2, 8], "unitari": [2, 8, 9], "termin": 2, "quantum": [2, 3, 7, 9, 10], "trajectori": [2, 6], "lead": [2, 5], "histori": 2, "reset": 2, "clear_final_st": 2, "If": 2, "your": [2, 8, 9], "you": [2, 3, 7, 8, 9, 10], "trajector": 2, "shown": [2, 3], "with_nois": 2, "depolar": 2, "05": 2, "The": [2, 3, 4, 5, 7, 8, 9], "simplest": 2, "krau": [2, 6], "depend": [2, 5, 8], "mixtur": 2, "kraus_oper": 2, "pair": [2, 6], "depolarizingchannel": 2, "01": 2, "99": 2, "arrai": [2, 7, 9], "0033333333333333335": 2, "j": [2, 3, 6, 7, 8, 9], "need": [2, 3, 5, 7, 8, 9, 10], "select": [2, 5, 6], "psi": [2, 3, 5, 7], "rangl": [2, 3, 7, 10], "k_i": 2, "p_i": 2, "langl": [2, 7, 10], "dagger": [2, 10], "noiseless": 2, "updat": [2, 3, 5, 6, 7, 8, 9], "evolut": [2, 8, 10], "isn": 2, "mani": [2, 3, 10], "differ": [2, 10], "should": [2, 8, 9, 10], "slower": 2, "manner": [2, 10], "accept": 2, "convert": [2, 5, 6], "most": [2, 5, 6, 10], "popular": 2, "specifi": [2, 4, 5, 8, 9], "qasm": 2, "string": [2, 5, 6, 10], "openqasm": 2, "qelib1": 2, "inc": 2, "qreg": 2, "q": [2, 3, 5, 9], "cx": 2, "contrib": [2, 5, 6], "qasm_import": 2, "circuit_from_qasm": 2, "q_0": 2, "q_1": 2, "algorithm": [3, 5, 7, 8, 10], "introduc": [3, 7], "without": [3, 7], "margin": [3, 7, 10], "phy": [3, 7], "rev": [3, 7], "lett": [3, 7], "hereaft": 3, "itertool": [3, 9], "consid": [3, 8], "three": [3, 7], "ghz": [3, 5], "nqubit": [3, 7, 8], "wai": 3, "wavefunct": [3, 7, 8, 9], "000": 3, "111": 3, "bit": [3, 8, 9, 10], "b_1": 3, "b_2": 3, "condit": 3, "being": [3, 6], "b_3": 3, "third": 3, "iter": [3, 9], "rho": 3, "outer": 3, "conj": 3, "reshap": [3, 7], "rho0": 3, "partial_trac": 3, "keep_indic": 3, "bit0": 3, "diag": 3, "real": 3, "rho1": 3, "bit1": 3, "rho2": 3, "bit2": 3, "join": 3, "after": [3, 5, 8, 9], "applic": [3, 8], "visual": [3, 5, 7], "walkthrough": 3, "ipython": 3, "displai": [3, 5], "youtubevideo": 3, "bkp96jijqku": 3, "width": [3, 5], "640": 3, "height": 3, "360": 3, "approach": 3, "otim": [3, 10], "n": [3, 5, 6, 9, 10], "loop": 3, "grab": 3, "determin": [3, 5], "its": [3, 5, 10], "all_oper": [3, 5], "qubit_to_index": 3, "enumer": [3, 5, 6], "0th": 3, "think": 3, "about": [3, 5], "place": [3, 7, 9], "wildcard": 3, "where": [3, 5, 6, 7, 8, 9, 10], "python": [3, 5, 6, 7], "candid": 3, "candidate_bitstr": 3, "product": [3, 4, 6, 8, 10], "choos": 3, "squar": 3, "get": [3, 4, 5, 6, 8, 9, 10], "simpli": [3, 5], "index": [3, 4, 9], "prob": 3, "ve": 3, "step": [3, 5, 8], "all_qubit": [3, 6], "core": [3, 5, 6], "produc": 3, "induct": 3, "proof": 3, "correct": 3, "bar": 3, "return_count": 3, "true": [3, 6, 10], "align": 3, "center": 3, "quick": 4, "detail": [4, 9], "short": 4, "answer": 4, "understand": 4, "custom": 4, "apply_op": [4, 7, 8], "featur": [4, 9], "access": 4, "noisi": 4, "intermedi": [4, 7], "qiskit": 4, "pennylan": 4, "etc": [4, 7, 9, 10], "refer": [4, 10], "tip": 4, "clifford": [4, 8, 10], "matrix": [4, 6, 7, 9, 10], "mp": [4, 5], "solv": 4, "qaoa": 4, "problem": 4, "modul": 4, "search": [4, 6], "page": 4, "class": [4, 5, 6, 9], "callabl": 4, "float": [4, 5, 6, 7], "program": [4, 6], "abstractcircuit": [4, 6], "param_resolv": [4, 6], "paramresolverorsimilartyp": 4, "mode": 4, "outcom": 4, "It": 4, "underli": 4, "mechan": 4, "paramet": [4, 6], "attribut": 4, "sympi": [4, 6], "symbol": [4, 6], "within": 4, "execut": [4, 7], "though": 4, "instead": [4, 5, 8], "rather": [4, 5, 9], "arg": [4, 5, 6, 9], "contain": [4, 8], "quimb": [5, 6], "mpsstate": [5, 6], "ccq": [5, 6], "mps_simul": [5, 6], "tensor": [5, 6, 7, 10], "qtn": [5, 6], "ignor": [5, 9], "call": [5, 6, 8], "ctype": 5, "callback": 5, "executionengin": 5, "_raw_object_cache_notifi": 5, "0x7f4195bd20c0": 5, "traceback": [5, 6], "recent": [5, 6], "last": [5, 6, 7, 9], "file": [5, 6], "opt": [5, 6], "hostedtoolcach": [5, 6], "x64": [5, 6], "lib": [5, 6], "python3": [5, 6], "site": [5, 6], "packag": [5, 6, 7], "llvmlite": 5, "bind": 5, "py": [5, 6], "line": [5, 6], "171": 5, "self": [5, 6, 9], "data": [5, 6], "keyboardinterrupt": [5, 6], "cirq_mps_bitstring_prob": [5, 6], "binari": [5, 6], "m_subset": [5, 6], "ai": [5, 6], "qubit_index": [5, 6], "i_str": [5, 6], "compon": [5, 6], "a_subset": [5, 6], "isel": [5, 6], "tensor_network": [5, 6], "tensornetwork": [5, 6], "contract": [5, 6], "inplac": [5, 6], "fals": [5, 6], "repres": [5, 7], "sum_": [5, 10], "tr": 5, "a_1": 5, "s_1": 5, "a_n": 5, "s_n": 5, "s_i": 5, "a_i": 5, "order": 5, "chi": 5, "entangl": 5, "system": [5, 6], "retriev": 5, "inform": 5, "involv": 5, "represent": [5, 7, 9, 10], "local": 5, "network": [5, 7, 10], "http": 5, "readthedoc": 5, "io": 5, "en": 5, "latest": 5, "html": 5, "larg": [5, 6, 10], "sequenc": [5, 6], "size": [5, 9], "arang": 5, "shuffl": 5, "6": [5, 6, 8], "7": [5, 6, 8], "qubit_fronti": 5, "circuit_to_tensor": 5, "tn": 5, "draw": [5, 6], "figsiz": 5, "discuss": [5, 7], "github": 5, "com": 5, "quantumlib": 5, "blob": 5, "master": 5, "ipynb": 5, "read": [5, 10], "qi": 5, "q0": 5, "open": 5, "leg": 5, "ij": 5, "qq": 5, "l": [5, 6, 7, 9], "lth": 5, "present": [5, 7], "left": 5, "right": 5, "orang": 5, "sequenti": 5, "between": [5, 8], "remain": 5, "onc": [5, 7], "incorpor": 5, "longer": 5, "becom": 5, "avail": 5, "emerg": 5, "tn2": 5, "integr": 5, "nativ": 5, "subsequ": 5, "backend": 5, "simplifi": 5, "i_j": 5, "jth": 5, "node": [5, 6], "act": [5, 9, 10], "i_0": 5, "subsequent": 5, "mps_state": 5, "prng": [5, 6], "mps_tn": 5, "primari": [5, 7], "thing": 5, "variou": 5, "examin": 5, "properti": 5, "observ": 5, "dimens": 5, "anoth": 5, "peripher": 5, "addit": [5, 8], "axi": [5, 6, 8], "extra": 5, "wa": 5, "ind": 5, "mu_0_3": 5, "mu_0_1": 5, "mu_0_5": 5, "mu_0_2": 5, "mu_0_7": 5, "mu_0_4": 5, "mu_0_6": 5, "tag": 5, "oset": 5, "i_1": 5, "i_2": 5, "i_3": 5, "i_4": 5, "i_5": 5, "i_6": 5, "i_7": 5, "A": [5, 7, 9], "naiv": 5, "techniqu": 5, "major": 5, "insight": 5, "elimin": 5, "whether": [5, 9], "requisit": 5, "up": 5, "11111111": 5, "turn": 5, "scalar": 5, "full": [5, 10], "m_tn": 5, "m_amp": 5, "4999999999999998": 5, "proper": 5, "mps_sim": 5, "500": 5, "hist": 5, "let": [5, 10], "oppos": 5, "14": 5, "22": [5, 8], "statevec_tim": 5, "mps_time": 5, "vec_sim": 5, "12": [5, 8], "18": 5, "notic": [5, 7], "quit": 5, "poor": 5, "yet": 5, "farili": 5, "ones": 5, "explain": 5, "global": 5, "simplif": 5, "connect": [5, 6], "maxim": [5, 6], "seek": 5, "improv": [5, 8], "lowli": 5, "subset": 5, "30": 5, "numcnot": 5, "drive": 5, "home": 5, "unentangl": 5, "constant": 5, "despit": 5, "continu": [5, 6, 9], "fullsiz": 5, "16": 5, "amount": [5, 8], "modest": 5, "linear": 5, "numgat": 5, "24": 5, "28": [5, 8], "spars": 6, "graph": 6, "particularli": 6, "primarili": 6, "degre": 6, "connected": 6, "maxcut": 6, "networkx": 6, "nx": 6, "svg": 6, "svgcircuit": 6, "bitstring_amplitud": 6, "rand_graph": 6, "erdos_renyi_graph": 6, "with_label": 6, "hamiltonian": 6, "translat": 6, "h_c": 6, "frac": 6, "z_iz_j": 6, "parametr": 6, "u_c": 6, "gamma": 6, "mix": 6, "h_m": 6, "x_i": 6, "u_m": 6, "beta": 6, "config_energi": 6, "assign": 6, "energi": 6, "edg": 6, "obj_func": 6, "averag": [6, 8], "uc": 6, "zz": 6, "um": 6, "construct_circuit": 6, "p_rang": 6, "g1": 6, "b1": 6, "maxcut_circuit": 6, "findfont": 6, "font": 6, "famili": 6, "arial": 6, "find": 6, "combin": [6, 9], "repeatedli": 6, "minim": 6, "run_sweep": 6, "ngamma": 6, "nbeta": 6, "param_sweep": 6, "9": 6, "length": 6, "lowchi": 6, "mpsoption": 6, "max_bond": 6, "bgls_mps_sampler": 6, "simulation_opt": 6, "param": 6, "cell": [6, 9], "72": 6, "simulatessampl": 6, "69": 6, "70": 6, "sweepabl": 6, "run_sweep_it": 6, "103": 6, "record": 6, "measurement_key_nam": 6, "empti": [6, 9], "102": 6, "_run": 6, "104": 6, "105": 6, "106": 6, "studi": 6, "125": 6, "122": 6, "paramresolv": 6, "123": 6, "resolved_circuit": 6, "resolve_paramet": 6, "_sampl": 6, "148": 6, "143": 6, "keys_to_bitstrings_list": 6, "145": 6, "needs_trajectori": 6, "146": 6, "pass": [6, 8], "147": 6, "_sample_from_one_wavefunction_evolut": 6, "149": 6, "150": 6, "151": 6, "152": 6, "153": 6, "154": 6, "219": 6, "215": 6, "_apply_op": 6, "217": 6, "skip": [6, 8], "thei": 6, "do": [6, 7, 8, 9, 10], "chang": [6, 8], "218": 6, "is_diagon": 6, "220": 6, "222": 6, "memoiz": 6, "_compute_prob": 6, "kraus_protocol": 6, "val": 6, "default": 6, "tupl": 6, "u": [6, 10], "mixture_result": 6, "unitary_gett": 6, "getattr": 6, "_unitary_": 6, "unitary_result": 6, "notimpl": 6, "155": 6, "156": 6, "gate_oper": 6, "195": 6, "gateoper": [6, 7], "193": 6, "getter": 6, "194": 6, "196": 6, "eigen_g": 6, "342": 6, "eigeng": 6, "340": 6, "341": 6, "cast": 6, "_expon": 6, "343": 6, "344": 6, "1j": 6, "half_turn": 6, "_global_shift": 6, "345": 6, "_eigen_compon": 6, "346": 6, "347": 6, "348": 6, "fromnumer": 6, "2313": 6, "keepdim": 6, "2310": 6, "2311": 6, "_wrapreduct": 6, "add": 6, "2314": 6, "88": 6, "obj": 6, "ufunc": 6, "kwarg": 6, "85": 6, "86": 6, "reduct": 6, "passkwarg": 6, "reduc": 6, "combo": [6, 9], "minimum": 6, "zero": [6, 9], "max_energi": 6, "max_param": 6, "pcolormesh": 6, "shade": 6, "nearest": 6, "colorbar": 6, "text": [6, 10], "2142857142857143": 6, "configur": 6, "solut": 6, "result_at_max": 6, "best_energi": 6, "max_best_energi": 6, "best_assign": 6, "best": 6, "slice": 6, "nodesa": 6, "nodesb": 6, "subset_color": 6, "red": [6, 8], "green": 6, "add_nod": 6, "node_for_ad": 6, "attr": 6, "add_edg": 6, "node_color": 6, "ravyi": 7, "osset": 7, "iu": 7, "ampl": 7, "instal": 7, "pip": 7, "x27": 7, "ingredi": 7, "densiti": [7, 10], "calcul": 7, "typic": [7, 8], "purpos": 7, "easier": [7, 10], "complex64": 7, "round": 7, "707": [7, 8], "must": 7, "modifi": 7, "probability_of_bitstr": 7, "2f": 7, "50": [7, 8], "ident": [7, 8], "reason": 7, "drawn": 7, "easili": 7, "come": 7, "situat": 7, "design": 8, "flexibl": 8, "fast": 8, "box": 8, "automat": 8, "resampl": 8, "On": 8, "top": 8, "few": 8, "user": 8, "tqdm": 8, "optimize_for_bgl": 8, "merg": 8, "optimized_circuit": 8, "want": 8, "bell": [8, 9], "four": 8, "six": 8, "action": 8, "word": 8, "k": [8, 10], "drop": 8, "matric": [8, 9, 10], "long": [8, 9], "structur": 8, "never": 8, "benchmark": 8, "all_tim": 8, "all_times_with_merged_op": 8, "desc": 8, "times_with_merged_op": 8, "circuit_optim": 8, "monoton": 8, "07": 8, "60": 8, "13": 8, "47": 8, "75": 8, "57": 8, "08": 8, "times_std": 8, "std": 8, "ddof": 8, "times_with_merged_ops_std": 8, "errorbar": 8, "yerr": 8, "capsiz": 8, "w": 8, "ratio": 8, "5x": 8, "2x": 8, "mention": 9, "guid": [9, 10], "simpl": 9, "__init__": 9, "num_qubit": 9, "complex": 9, "copi": 9, "new_stat": 9, "instanti": 9, "abl": 9, "assum": [9, 10], "aren": 9, "takeawai": 9, "plug": 9, "safe": 9, "skim": 9, "own": 9, "d": [9, 10], "write": 9, "apply_matrix_g": 9, "target_bit": 9, "qu": 9, "indic": 9, "flag": 9, "already_appli": 9, "subspace_indic": 9, "powerset": 9, "i0": 9, "i1": 9, "element": 9, "tempidx": 9, "flip_bit": 9, "apply_op_to_subspac": 9, "asarrai": 9, "idx": 9, "subspac": 9, "integ": 9, "dot": 9, "chain": 9, "from_iter": 9, "bit_to_flip": 9, "th": 9, "flip": 9, "intend": 9, "prepar": 9, "70710678": 9, "format": 9, "sort": 9, "samplign": 9, "discus": 10, "memori": 10, "limit": 10, "classic": 10, "summari": 10, "bravyi": 10, "gosset": 10, "liu": 10, "argument": 10, "why": 10, "evolv": 10, "dure": 10, "u_t": 10, "supp": 10, "u_d": 10, "cdot": 10, "u_1": 10, "quantiti": 10, "upper": 10, "bound": 10, "max_t": 10, "denot": 10, "2d": 10, "sai": 10, "sens": 10, "schroding": 10, "2n": 10, "feynman": 10, "unlik": 10, "signific": 10, "specif": 10, "illustr": 10}, "objects": {"bgls": [[4, 0, 1, "", "Simulator"]], "bgls.Simulator": [[4, 1, 1, "", "run"]]}, "objtypes": {"0": "py:class", "1": "py:method"}, "objnames": {"0": ["py", "class", "Python class"], "1": ["py", "method", "Python method"]}, "titleterms": {"clifford": [0, 1], "decomposit": 0, "us": [1, 2, 5, 6, 7, 8, 9, 10], "bgl": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], "circuit": [1, 2, 5, 8], "featur": 2, "access": 2, "final": 2, "state": [2, 5, 9], "simul": [2, 4, 7], "noisi": 2, "how": [2, 3, 7, 9], "modifi": 2, "apply_op": [2, 9], "support": 2, "channel": 2, "intermedi": 2, "measur": 2, "can": 2, "i": 2, "qiskit": 2, "pennylan": 2, "etc": 2, "work": 3, "qubit": [3, 10], "sampl": [3, 5, 10], "gate": [3, 10], "video": 3, "explain": 3, "discuss": 3, "exampl": [3, 4, 10], "put": 3, "all": 3, "togeth": 3, "refer": 3, "document": 4, "guid": 4, "indic": 4, "tabl": 4, "from": 5, "matrix": 5, "product": 5, "run": 5, "mp": 6, "solv": 6, "qaoa": 6, "problem": 6, "get": 7, "start": 7, "quick": 7, "more": 7, "detail": 7, "creat": 7, "when": [7, 10], "tip": 8, "optim": 8, "tl": 8, "dr": 8, "warn": 8, "Will": 8, "preserv": 8, "gateset": 8, "explan": 8, "time": 8, "test": 8, "custom": 9, "setup": 9, "defin": 9, "compute_prob": 9, "The": 10, "short": 10, "answer": 10, "understand": 10, "cost": 10, "ratio": 10}, "envversion": {"sphinx.domains.c": 2, "sphinx.domains.changeset": 1, "sphinx.domains.citation": 1, "sphinx.domains.cpp": 8, "sphinx.domains.index": 1, "sphinx.domains.javascript": 2, "sphinx.domains.math": 2, "sphinx.domains.python": 3, "sphinx.domains.rst": 2, "sphinx.domains.std": 2, "sphinx": 57}, "alltitles": {"Clifford Decomposition": [[0, "clifford-decomposition"]], "Using BGLS on Clifford and near-Clifford circuits": [[1, "using-bgls-on-clifford-and-near-clifford-circuits"]], "Features": [[2, "features"]], "Accessing final states": [[2, "accessing-final-states"]], "Simulating noisy circuits": [[2, "simulating-noisy-circuits"]], "How to modify apply_op to support channels": [[2, "how-to-modify-apply-op-to-support-channels"]], "Simulating circuits with intermediate measurements": [[2, "simulating-circuits-with-intermediate-measurements"]], "Can I use BGLS with Qiskit / Pennylane / etc.?": [[2, "can-i-use-bgls-with-qiskit-pennylane-etc"]], "How BGLS works": [[3, "how-bgls-works"]], "Qubit-by-qubit sampling": [[3, "qubit-by-qubit-sampling"], [10, "qubit-by-qubit-sampling"]], "Gate-by-gate sampling": [[3, "gate-by-gate-sampling"]], "Video explainer": [[3, "video-explainer"]], "Discussion and example": [[3, "discussion-and-example"]], "Putting it all together": [[3, "putting-it-all-together"]], "References": [[3, "references"]], "Documentation for BGLS": [[4, "documentation-for-bgls"]], "Guide": [[4, null]], "Examples": [[4, null], [10, "examples"]], "Indices and tables": [[4, "indices-and-tables"]], "Bgls Simulator": [[4, "bgls-simulator"]], "Using BGLS to sample from Matrix Product State Circuits": [[5, "using-bgls-to-sample-from-matrix-product-state-circuits"]], "Running on BGLS": [[5, "running-on-bgls"]], "Using BGLS with MPS to solve QAOA problems": [[6, "using-bgls-with-mps-to-solve-qaoa-problems"]], "Getting started": [[7, "getting-started"]], "Quick start": [[7, "quick-start"]], "More detail: How to create a bgls.Simulator": [[7, "more-detail-how-to-create-a-bgls-simulator"]], "When to use the bgls.Simulator": [[7, "when-to-use-the-bgls-simulator"]], "Tips for using BGLS": [[8, "tips-for-using-bgls"]], "Optimizing circuits": [[8, "optimizing-circuits"]], "TL;DR": [[8, "tl-dr"]], "Warning: Will not preserve gateset": [[8, "warning-will-not-preserve-gateset"]], "Explanation": [[8, "explanation"]], "Timing test": [[8, "timing-test"]], "Using custom states with BGLS": [[9, "using-custom-states-with-bgls"]], "Setup": [[9, "setup"]], "Defining a custom state": [[9, "defining-a-custom-state"]], "Defining how to apply_ops": [[9, "defining-how-to-apply-ops"]], "Defining how to compute_probability": [[9, "defining-how-to-compute-probability"]], "Using with BGLS": [[9, "using-with-bgls"]], "When to use BGLS": [[10, "when-to-use-bgls"]], "The short answer": [[10, "the-short-answer"]], "Understanding the cost": [[10, "understanding-the-cost"]], "BGLS gate-by-gate sampling": [[10, "bgls-gate-by-gate-sampling"]], "The cost ratio": [[10, "the-cost-ratio"]]}, "indexentries": {"simulator (class in bgls)": [[4, "bgls.Simulator"]], "run() (bgls.simulator method)": [[4, "bgls.Simulator.run"]]}})