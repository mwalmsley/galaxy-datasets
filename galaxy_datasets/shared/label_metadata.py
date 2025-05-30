

# use these by importing them in another script e.g.
# from galaxy_datasets.shared.label_metadata import decals_label_cols 

# The "ortho" versions avoid mixing votes from different campaigns by appending -{campaign} (e.g. -dr5, -gz2) to each question

# includes DECaLS, GZ2, GZ Rings, GZ Cosmic Dawn


def extract_questions_and_label_cols(question_answer_pairs):
    """
    Convenience wrapper to get list of questions and label_cols from a schema.
    Common starting point for analysis, iterating over questions, etc.

    Args:
        question_answer_pairs (dict): e.g. {'smooth-or-featured: ['_smooth, _featured-or-disk, ...], ...}

    Returns:
        list: all questions e.g. [Question('smooth-or-featured'), ...]
        list: label_cols (list of answer strings). See ``label_metadata.py`` for examples.
    """
    questions = list(question_answer_pairs.keys())
    label_cols = [q + answer for q, answers in question_answer_pairs.items() for answer in answers]
    return questions, label_cols



"""
DECALS (GZD-1, GZD-2, GZD-5, GZD-8)
"""

# the schema is slightly different for dr1/2 vs dr5+
# merging answers were changed completely
# bulge sizes were changed from 3 to 5
# bar was changed from yes/no to strong/weak/none
# spiral had 'cant tell' added
decals_dr12_ortho_pairs = {
    'smooth-or-featured-dr12': ['_smooth', '_featured-or-disk', '_artifact'],
    'disk-edge-on-dr12': ['_yes', '_no'],
    'has-spiral-arms-dr12': ['_yes', '_no'],
    'bar-dr12': ['_yes', '_no'],
    'bulge-size-dr12': ['_dominant', '_obvious', '_none'],
    'how-rounded-dr12': ['_completely', '_in-between', '_cigar-shaped'],  # completely was renamed to round
    'edge-on-bulge-dr12': ['_boxy', '_none', '_rounded'],
    'spiral-winding-dr12': ['_tight', '_medium', '_loose'],
    'spiral-arm-count-dr12': ['_1', '_2', '_3', '_4', '_more-than-4'],
    'merging-dr12': ['_neither', '_tidal-debris', '_both', '_merger']
}

decals_dr5_ortho_pairs = {
    'smooth-or-featured-dr5': ['_smooth', '_featured-or-disk', '_artifact'],
    'disk-edge-on-dr5': ['_yes', '_no'],
    'has-spiral-arms-dr5': ['_yes', '_no'],
    'bar-dr5': ['_strong', '_weak', '_no'],
    'bulge-size-dr5': ['_dominant', '_large', '_moderate', '_small', '_none'],
    'how-rounded-dr5': ['_round', '_in-between', '_cigar-shaped'],
    'edge-on-bulge-dr5': ['_boxy', '_none', '_rounded'],
    'spiral-winding-dr5': ['_tight', '_medium', '_loose'],
    'spiral-arm-count-dr5': ['_1', '_2', '_3', '_4', '_more-than-4', '_cant-tell'],
    'merging-dr5': ['_none', '_minor-disturbance', '_major-disturbance', '_merger']
}

decals_dr5_ortho_questions, decals_dr5_ortho_label_cols = extract_questions_and_label_cols(decals_dr5_ortho_pairs)

# exactly the same for dr8. 
decals_dr8_ortho_pairs = decals_dr5_ortho_pairs.copy()
for question, answers in decals_dr8_ortho_pairs.copy().items(): # avoid modifying while looping
    decals_dr8_ortho_pairs[question.replace('-dr5', '-dr8')] = answers
    del decals_dr8_ortho_pairs[question]  # delete the old ones


# I think performance should be best when training on *both*
decals_all_campaigns_ortho_pairs = {}  # big dict including all the pairs from DR1/2, DR5, DR8
decals_all_campaigns_ortho_pairs.update(decals_dr12_ortho_pairs)
decals_all_campaigns_ortho_pairs.update(decals_dr5_ortho_pairs)
decals_all_campaigns_ortho_pairs.update(decals_dr8_ortho_pairs)

# for convenience, extract lists of just the questions and just the label columns
# these can then be imported throughout without needing to apply this function elsewhere
decals_dr12_ortho_questions, decals_dr12_ortho_label_cols = extract_questions_and_label_cols(decals_dr12_ortho_pairs)
decals_dr5_ortho_questions, decals_dr5_ortho_label_cols = extract_questions_and_label_cols(decals_dr5_ortho_pairs)
decals_dr8_ortho_questions, decals_dr8_ortho_label_cols = extract_questions_and_label_cols(decals_dr8_ortho_pairs)
decals_all_campaigns_ortho_questions, decals_all_campaigns_ortho_label_cols = extract_questions_and_label_cols(decals_all_campaigns_ortho_pairs)


# Dict mapping each question (e.g. disk-edge-on) 
# to the answer on which it depends (e.g. smooth-or-featured_featured-or-disk)
decals_ortho_dependencies = {
    # dr12
    'smooth-or-featured-dr12': None,
    'disk-edge-on-dr12': 'smooth-or-featured-dr12_featured-or-disk',
    'has-spiral-arms-dr12': 'disk-edge-on-dr12_no',
    'bar-dr12': 'disk-edge-on-dr12_no',
    'bulge-size-dr12': 'disk-edge-on-dr12_no',
    'how-rounded-dr12': 'smooth-or-featured-dr12_smooth',
    'edge-on-bulge-dr12': 'disk-edge-on-dr12_yes',
    'spiral-winding-dr12': 'has-spiral-arms-dr12_yes',
    'spiral-arm-count-dr12': 'has-spiral-arms-dr12_yes',
    'merging-dr12': None,  # TODO technically should be smooth OR featured, but hard to code - no double dependency support yet
    # dr5
    'smooth-or-featured-dr5': None,  # always asked
    'disk-edge-on-dr5': 'smooth-or-featured-dr5_featured-or-disk',
    'has-spiral-arms-dr5': 'disk-edge-on-dr5_no',
    'bar-dr5': 'disk-edge-on-dr5_no',
    'bulge-size-dr5': 'disk-edge-on-dr5_no',
    'how-rounded-dr5': 'smooth-or-featured-dr5_smooth',
    'edge-on-bulge-dr5': 'disk-edge-on-dr5_yes',
    'spiral-winding-dr5': 'has-spiral-arms-dr5_yes',
    'spiral-arm-count-dr5': 'has-spiral-arms-dr5_yes', # bad naming...
    'merging-dr5': None,
    # dr8 is identical to dr5, just with -dr8
    'smooth-or-featured-dr8': None,
    'disk-edge-on-dr8': 'smooth-or-featured-dr8_featured-or-disk',
    'has-spiral-arms-dr8': 'disk-edge-on-dr8_no',
    'bar-dr8': 'disk-edge-on-dr8_no',
    'bulge-size-dr8': 'disk-edge-on-dr8_no',
    'how-rounded-dr8': 'smooth-or-featured-dr8_smooth',
    'edge-on-bulge-dr8': 'disk-edge-on-dr8_yes',
    'spiral-winding-dr8': 'has-spiral-arms-dr8_yes',
    'spiral-arm-count-dr8': 'has-spiral-arms-dr8_yes',
    'merging-dr8': None,
    }


"""
DESI

Identical to DECalS DR5, DR8, except without the -dr8
Not used for ML, only for user convenience
"""

desi_pairs = decals_dr8_ortho_pairs.copy()
for question, answers in desi_pairs.copy().items(): # avoid modifying while looping
    desi_pairs[question.replace('-dr8', '')] = answers
    del desi_pairs[question]  # delete the old ones

desi_dependencies = {}
for question, dependency in decals_ortho_dependencies.copy().items():
    if '-dr8' in question:
        question_text = question.replace('-dr8', '')
        if dependency is None:
            dependency_text = None
        else:
            dependency_text = dependency.replace('-dr8', '')
        desi_dependencies[question_text] = dependency_text 


"""
Galaxy Zoo 2 (GZ2)
"""

gz2_ortho_pairs = {
    'smooth-or-featured-gz2': ['_smooth', '_featured-or-disk', '_artifact'],
    'disk-edge-on-gz2': ['_yes', '_no'],
    'has-spiral-arms-gz2': ['_yes', '_no'],
    'bar-gz2': ['_yes', '_no'],
    'bulge-size-gz2': ['_dominant', '_obvious', '_just-noticeable', '_no'],
    'something-odd-gz2': ['_yes', '_no'],
    'how-rounded-gz2': ['_round', '_in-between', '_cigar'],
    'bulge-shape-gz2': ['_round', '_boxy', '_no-bulge'],
    'spiral-winding-gz2': ['_tight', '_medium', '_loose'],
    'spiral-arm-count-gz2': ['_1', '_2', '_3', '_4', '_more-than-4', '_cant-tell']
}
gz2_ortho_questions, gz2_ortho_label_cols = extract_questions_and_label_cols(gz2_ortho_pairs)

gz2_ortho_dependencies = {
    'smooth-or-featured-gz2': None,  # always asked
    'disk-edge-on-gz2': 'smooth-or-featured-gz2_featured-or-disk',
    'has-spiral-arms-gz2': 'smooth-or-featured-gz2_featured-or-disk',
    'bar-gz2': 'smooth-or-featured-gz2_featured-or-disk',
    'bulge-size-gz2': 'smooth-or-featured-gz2_featured-or-disk',
    'how-rounded-gz2': 'smooth-or-featured-gz2_smooth',
    'bulge-shape-gz2': 'disk-edge-on-gz2_yes',  # gz2 only
    'edge-on-bulge-gz2': 'disk-edge-on-gz2_yes',
    'spiral-winding-gz2': 'has-spiral-arms-gz2_yes',
    'spiral-arm-count-gz2': 'has-spiral-arms-gz2_yes',
    'something-odd-gz2': None  # actually neglects the artifact branch
}

"""
Galaxy Zoo Rings (GZ Rings)
"""

rings_pairs = {
    'ring': ['_yes', '_no']
}
rings_questions, rings_label_cols = extract_questions_and_label_cols(rings_pairs)

rings_dependencies = {'ring': None }


"""
Galaxy Zoo Hubble (and Euclidized)
"""

# TODO may change features to featured-or-disk
hubble_pairs = {
    'smooth-or-featured': ['_smooth', '_features', '_artifact'],
    'how-rounded': ['_completely', '_in-between', '_cigar-shaped'],
    'clumpy-appearance': ['_yes', '_no'],
    # 'clump-count': ['_1', '_2', '_3', '_4', '_5-plus', '_cant-tell'],
    # disable these for now as I don't support having several but not all answers leading to the same next question
    # 'clump-configuration': ['_straight-line', '_chain', '_cluster-or-irregular', '_spiral'],
    # 'one-clump-brightest': ['_yes', '_no'],
    # 'brightest-clump-central': ['_yes', '_no'],
    'disk-edge-on': ['_yes', '_no'],
    'bulge-shape': ['_rounded', '_boxy', '_none'],
    'bar': ['_yes', '_no'],
    'has-spiral-arms': ['_yes', '_no'],
    'spiral-winding': ['_tight', '_medium', '_loose'],
    'spiral-arm-count': ['_1', '_2', '_3', '_4', '_5-plus', '_cant-tell'],
    'bulge-size': ['_none', '_just-noticeable', '_obvious', '_dominant'],
    'galaxy-symmetrical': ['_yes', '_no'],
    'clumps-embedded-larger-object': ['_yes', '_no']
}
# add -hubble to the end of each question
hubble_ortho_pairs = dict([(key + '-hubble', value) for key, value in hubble_pairs.items()])

# not used here, but may be helpful elsewhere
hubble_ortho_dependencies = {
    'smooth-or-featured-hubble': None,
    'how-rounded-hubble': 'smooth-or-featured-hubble_smooth',
    'clumpy-appearance-hubble': 'smooth-or-featured-hubble_features',
    # 'clump-count-hubble': 'clumpy-appearance-hubble_yes',
    # 'clump-configuration-hubble': ['_straight-line', '_chain', '_cluster-or-irregular', '_spiral'],
    # 'one-clump-brightest-hubble': ['_yes', '_no'],
    # 'brightest-clump-central-hubble': ['_yes', '_no'],
    # ignoring the spiral dashed line, probably rare
    'galaxy-symmetrical-hubble': 'clumpy-appearance-hubble_yes',
    'clumps-embedded-larger-object-hubble': 'clumpy-appearance-hubble_yes',
    'disk-edge-on-hubble': 'clumpy-appearance-hubble_no',
    'bulge-shape-hubble': 'disk-edge-on-hubble_yes',
    'edge-on-bulge-hubble': 'disk-edge-on-hubble_yes',
    'bar-hubble': 'disk-edge-on-hubble_no',
    'has-spiral-arms-hubble': 'disk-edge-on-hubble_no',
    'spiral-winding-hubble': 'disk-edge-on-hubble_no',
    'spiral-arm-count-hubble': 'disk-edge-on-hubble_no',
    'bulge-size-hubble': 'disk-edge-on-hubble_no'
}

hubble_ortho_questions, hubble_ortho_label_cols = extract_questions_and_label_cols(hubble_ortho_pairs)



# similar but simplified version avoiding the questionable clump questions
# Courtesy David O'Ryan - thanks, David!
hubble_v2_pairs = {
    'smooth-or-featured': ['_smooth', '_features', '_artifact'],
    'how-rounded': ['_completely', '_in-between', '_cigar-shaped'],
    'disk-edge-on': ['_yes', '_no'],
    'bulge-shape': ['_rounded', '_boxy', '_none'],
    'bar': ['_yes', '_no'],
    'has-spiral-arms': ['_yes', '_no'],
    'spiral-winding': ['_tight', '_medium', '_loose'],
    'spiral-arm-count': ['_1', '_2', '_3', '_4', '_5-plus', '_cant-tell'],
    'bulge-size': ['_none', '_just-noticeable', '_obvious', '_dominant'],
}
hubble_v2_ortho_pairs = dict([(key + '-hubble', value) for key, value in hubble_v2_pairs.items()])

hubble_v2_ortho_dependencies = {
    'smooth-or-featured-hubble': None,
    'how-rounded-hubble': 'smooth-or-featured-hubble_smooth',
    'disk-edge-on-hubble': 'smooth-or-featured-hubble_features',
    'bulge-shape-hubble': 'disk-edge-on-hubble_yes',
    'edge-on-bulge-hubble': 'disk-edge-on-hubble_yes',
    'bar-hubble': 'disk-edge-on-hubble_no',
    'has-spiral-arms-hubble': 'disk-edge-on-hubble_no',
    'spiral-winding-hubble': 'disk-edge-on-hubble_no',
    'spiral-arm-count-hubble': 'disk-edge-on-hubble_no',
    'bulge-size-hubble': 'disk-edge-on-hubble_no'
}
hubble_v2_ortho_questions, hubble_v2_ortho_label_cols = extract_questions_and_label_cols(hubble_v2_ortho_pairs)

"""
Galaxy Zoo CANDELS
"""


# TODO may change features to featured-or-disk
candels_pairs = {
    'smooth-or-featured': ['_smooth', '_features', '_artifact'],
    'how-rounded': ['_completely', '_in-between', '_cigar-shaped'],
    'clumpy-appearance': ['_yes', '_no'],
    # 'clump-count': ['_1', '_2', '_3', '_4', '_5-plus', '_cant-tell'],
    # disable these for now as I don't support having several but not all answers leading to the same next question
    # 'clump-configuration': ['_straight-line', '_chain', '_cluster-or-irregular', '_spiral'],
    # 'one-clump-brightest': ['_yes', '_no'],
    # 'brightest-clump-central': ['_yes', '_no'],
    # 'galaxy-symmetrical': ['_yes', '_no'],
    # 'clumps-embedded-larger-object': ['_yes', '_no'],
    'disk-edge-on': ['_yes', '_no'],
    'edge-on-bulge': ['_yes', '_no'],
    'bar': ['_yes', '_no'],
    'has-spiral-arms': ['_yes', '_no'],
    'spiral-winding': ['_tight', '_medium', '_loose'],
    'spiral-arm-count': ['_1', '_2', '_3', '_4', '_5-plus', '_cant-tell'],
    'bulge-size': ['_none', '_obvious', '_dominant'],
    'merging': ['_merger', '_tidal-debris', '_both', '_neither']
}
# add -candels to the end of each question
candels_ortho_pairs = dict([(key + '-candels', value) for key, value in candels_pairs.items()])

# not used here, but may be helpful elsewhere
candels_ortho_dependencies = {
    'smooth-or-featured-candels': None,
    'how-rounded-candels': 'smooth-or-featured-candels_smooth',
    'clumpy-appearance-candels': 'smooth-or-featured-candels_features',
    # 'clump-count-candels': 'clumpy-appearance-candels_yes',
    # 'clump-configuration-candels': ['_straight-line', '_chain', '_cluster-or-irregular', '_spiral'],
    # 'one-clump-brightest-candels': ['_yes', '_no'],
    # 'brightest-clump-central-candels': ['_yes', '_no'],
    # 'galaxy-symmetrical-candels': ['_yes', '_no'],
    # 'clumps-embedded-larger-object-candels': ['_yes', '_no'],
    'disk-edge-on-candels': 'clumpy-appearance-candels_no',
    'edge-on-bulge-candels': 'disk-edge-on-candels_yes',
    'bar-candels': 'disk-edge-on-candels_no',
    'has-spiral-arms-candels': 'disk-edge-on-candels_no',
    'spiral-winding-candels': 'disk-edge-on-candels_no',
    'spiral-arm-count-candels': 'disk-edge-on-candels_no',
    'bulge-size-candels': 'disk-edge-on-candels_no',
    'merging-candels': None
}

candels_ortho_questions, candels_ortho_label_cols = extract_questions_and_label_cols(candels_ortho_pairs)


"""
Galaxy Zoo Cosmic Dawn (HSC)

Not yet created in galaxy-datasets
"""


cosmic_dawn_ortho_pairs = {
    'smooth-or-featured-cd': ['_smooth', '_featured-or-disk', '_problem'],  # renamed from artifact
    'disk-edge-on-cd': ['_yes', '_no'],
    'has-spiral-arms-cd': ['_yes', '_no'],
    'bar-cd': ['_strong', '_weak', '_no'],
    'bulge-size-cd': ['_dominant', '_large', '_moderate', '_small', '_none'],
    'how-rounded-cd': ['_round', '_in-between', '_cigar-shaped'],
    'edge-on-bulge-cd': ['_boxy', '_none', '_rounded'],
    'spiral-winding-cd': ['_tight', '_medium', '_loose'],
    'spiral-arm-count-cd': ['_1', '_2', '_3', '_4', '_more-than-4', '_cant-tell'],
    'merging-cd': ['_none', '_minor-disturbance', '_major-disturbance', '_merger'],
    'clumps-cd': ['_yes', '_no'],
    'problem-cd': ['_star', '_artifact', '_zoom'],
    'artifact-cd': ['_satellite', '_scattered', '_diffraction', '_ray', '_saturation', '_other']
}

cosmic_dawn_ortho_dependencies = {
    'smooth-or-featured-cd': None,
    'problem-cd': 'smooth-or-featured-cd_problem',  # newly added (new problem branch)
    'artifact-cd': 'problem-cd_artifact',  # newly added (new problem branch)
    'disk-edge-on-cd': 'smooth-or-featured-cd_featured-or-disk',
    'clumps-cd': 'disk-edge-on-cd_no',  # newly added (featured branch)
    'has-spiral-arms-cd': 'disk-edge-on-cd_no',
    'bar-cd': 'disk-edge-on-cd_no',
    'bulge-size-cd': 'disk-edge-on-cd_no',
    'how-rounded-cd': 'smooth-or-featured-cd_smooth',
    'edge-on-bulge-cd': 'disk-edge-on-cd_yes',
    'spiral-winding-cd': 'has-spiral-arms-cd_yes',
    'spiral-arm-count-cd': 'has-spiral-arms-cd_yes',
    'merging-cd': None  # technically, should be smooth OR featured, but hard to code
}

cosmic_dawn_ortho_questions, cosmic_dawn_ortho_label_cols = extract_questions_and_label_cols(cosmic_dawn_ortho_pairs)

def change_suffix(old_pairs, old_dependancies, old_suffix, new_suffix):
    pairs = old_pairs.copy()
    dependencies = {}

    for question, answers in pairs.copy().items(): # avoid modifying while looping
        pairs[question.replace(old_suffix, new_suffix)] = answers
        del pairs[question]  # delete the old ones

    for question, dependency in old_dependancies.copy().items():
        question_text = question.replace(old_suffix, new_suffix)
        if dependency is None:
            dependency_text = None
        else:
            dependency_text = dependency.replace(old_suffix, new_suffix)
        dependencies[question_text] = dependency_text 

    return pairs, dependencies


cosmic_dawn_pairs, cosmic_dawn_dependencies = change_suffix(cosmic_dawn_ortho_pairs, cosmic_dawn_ortho_dependencies, old_suffix='-cd', new_suffix='')

# jwst is exactly the same except with -jwst instead of -cd
# (has diffraction spikes under rare features, but we don't train on that anyway)
jwst_ortho_pairs, jwst_ortho_dependencies = change_suffix(cosmic_dawn_ortho_pairs, cosmic_dawn_ortho_dependencies, old_suffix='-cd', new_suffix='-jwst')
"""
UKIDSS (almost the same as GZ2)

"""

ukidss_ortho_pairs = {
    'smooth-or-featured-ukidss': ['_smooth', '_featured-or-disk', '_artifact'],
    'disk-edge-on-ukidss': ['_yes', '_no'],
    'has-spiral-arms-ukidss': ['_yes', '_no'],
    'bar-ukidss': ['_yes', '_no'],
    'bulge-size-ukidss': ['_dominant', '_obvious', '_just-noticeable', '_no'],
    'something-odd-ukidss': ['_yes', '_no'],
    'how-rounded-ukidss': ['_round', '_in-between', '_cigar'],
    'bulge-shape-ukidss': ['_round', '_boxy', '_no-bulge'],
    'spiral-winding-ukidss': ['_tight', '_medium', '_loose'],
    'spiral-arm-count-ukidss': ['_1', '_2', '_3', '_4', '_more-than-4', '_cant-tell']
}
ukidss_ortho_questions, ukidss_ortho_label_cols = extract_questions_and_label_cols(ukidss_ortho_pairs)

ukidss_ortho_dependencies = {
    'smooth-or-featured-ukidss': None,  # always asked
    'disk-edge-on-ukidss': 'smooth-or-featured-ukidss_featured-or-disk',
    'has-spiral-arms-ukidss': 'smooth-or-featured-ukidss_featured-or-disk',
    'bar-ukidss': 'smooth-or-featured-ukidss_featured-or-disk',
    'bulge-size-ukidss': 'smooth-or-featured-ukidss_featured-or-disk',
    'how-rounded-ukidss': 'smooth-or-featured-ukidss_smooth',
    'bulge-shape-ukidss': 'disk-edge-on-ukidss_yes',  # ukidss only
    'edge-on-bulge-ukidss': 'disk-edge-on-ukidss_yes',
    'spiral-winding-ukidss': 'has-spiral-arms-ukidss_yes',
    'spiral-arm-count-ukidss': 'has-spiral-arms-ukidss_yes',
    'something-odd-ukidss': None  # actually neglects the artifact branch
}


# same as cosmic dawn (except ghosts) for now
euclid_ortho_pairs = {
    'smooth-or-featured-euclid': ['_smooth', '_featured-or-disk', '_problem'],  # renamed from artifact
    'disk-edge-on-euclid': ['_yes', '_no'],
    'has-spiral-arms-euclid': ['_yes', '_no'],
    'bar-euclid': ['_strong', '_weak', '_no'],
    'bulge-size-euclid': ['_dominant', '_large', '_moderate', '_small', '_none'],
    'how-rounded-euclid': ['_round', '_in-between', '_cigar-shaped'],
    'edge-on-bulge-euclid': ['_boxy', '_none', '_rounded'],
    'spiral-winding-euclid': ['_tight', '_medium', '_loose'],
    'spiral-arm-count-euclid': ['_1', '_2', '_3', '_4', '_more-than-4', '_cant-tell'],
    'merging-euclid': ['_none', '_minor-disturbance', '_major-disturbance', '_merger'],
    'clumps-euclid': ['_yes', '_no'],
    'problem-euclid': ['_star', '_artifact', '_zoom'],
    'artifact-euclid': ['_satellite', '_scattered', '_diffraction', '_ray', '_saturation', '_other', '_ghost']
}

euclid_ortho_dependencies = {
    'smooth-or-featured-euclid': None,
    'problem-euclid': 'smooth-or-featured-euclid_problem',  # newly added (new problem branch)
    'artifact-euclid': 'problem-euclid_artifact',  # newly added (new problem branch)
    'disk-edge-on-euclid': 'smooth-or-featured-euclid_featured-or-disk',
    'clumps-euclid': 'disk-edge-on-euclid_no',  # newly added (featured branch)
    'has-spiral-arms-euclid': 'disk-edge-on-euclid_no',
    'bar-euclid': 'disk-edge-on-euclid_no',
    'bulge-size-euclid': 'disk-edge-on-euclid_no',
    'how-rounded-euclid': 'smooth-or-featured-euclid_smooth',
    'edge-on-bulge-euclid': 'disk-edge-on-euclid_yes',
    'spiral-winding-euclid': 'has-spiral-arms-euclid_yes',
    'spiral-arm-count-euclid': 'has-spiral-arms-euclid_yes',
    'merging-euclid': None  # technically, should be smooth OR featured, but hard to code
}

euclid_ortho_questions, euclid_ortho_label_cols = extract_questions_and_label_cols(euclid_ortho_pairs)

euclid_pairs, euclid_dependencies = change_suffix(euclid_ortho_pairs, euclid_ortho_dependencies, old_suffix='-euclid', new_suffix='')
euclid_questions, euclid_label_cols = extract_questions_and_label_cols(euclid_pairs)




def get_gz_evo_v1_metadata(internal):

    question_answer_pairs = {}
    question_answer_pairs.update(decals_all_campaigns_ortho_pairs)
    question_answer_pairs.update(gz2_ortho_pairs)
    question_answer_pairs.update(candels_ortho_pairs)
    question_answer_pairs.update(hubble_ortho_pairs)
    if internal:
        question_answer_pairs.update(cosmic_dawn_ortho_pairs)

    dependencies = {}
    dependencies.update(decals_ortho_dependencies)
    dependencies.update(gz2_ortho_dependencies)
    dependencies.update(candels_ortho_dependencies)
    dependencies.update(hubble_ortho_dependencies)
    if internal:
        dependencies.update(cosmic_dawn_ortho_dependencies)

    label_cols = \
        decals_all_campaigns_ortho_label_cols + \
        gz2_ortho_label_cols + \
        candels_ortho_label_cols + \
        hubble_ortho_label_cols
    if internal:
        label_cols += cosmic_dawn_ortho_label_cols

    return label_cols, question_answer_pairs, dependencies

gz_evo_v1_label_cols, gz_evo_v1_pairs, gz_evo_v1_dependencies = get_gz_evo_v1_metadata(internal=True)
gz_evo_v1_public_label_cols, gz_evo_v1_public_pairs, gz_evo_v1_public_dependencies = get_gz_evo_v1_metadata(internal=False)


""" same again for V2, adding Hubble V2 and Euclid"""
def get_gz_evo_v2_metadata(internal):

    question_answer_pairs = {}
    question_answer_pairs.update(decals_all_campaigns_ortho_pairs)
    question_answer_pairs.update(gz2_ortho_pairs)
    question_answer_pairs.update(candels_ortho_pairs)
    question_answer_pairs.update(hubble_v2_ortho_pairs)
    if internal:
        question_answer_pairs.update(cosmic_dawn_ortho_pairs)
        question_answer_pairs.update(euclid_ortho_pairs)

    dependencies = {}
    dependencies.update(decals_ortho_dependencies)
    dependencies.update(gz2_ortho_dependencies)
    dependencies.update(candels_ortho_dependencies)
    dependencies.update(hubble_v2_ortho_dependencies)
    if internal:
        dependencies.update(cosmic_dawn_ortho_dependencies)
        dependencies.update(euclid_ortho_dependencies)

    label_cols = \
        decals_all_campaigns_ortho_label_cols + \
        gz2_ortho_label_cols + \
        candels_ortho_label_cols + \
        hubble_v2_ortho_label_cols
    if internal:
        label_cols += cosmic_dawn_ortho_label_cols
        label_cols += euclid_ortho_label_cols

    return label_cols, question_answer_pairs, dependencies

gz_evo_v2_label_cols, gz_evo_v2_pairs, gz_evo_v2_dependencies = get_gz_evo_v2_metadata(internal=True)
gz_evo_v2_public_label_cols, gz_evo_v2_public_pairs, gz_evo_v2_public_dependencies = get_gz_evo_v2_metadata(internal=False)

# JWST CEERS 
jwst_ortho_pairs = {
    'smooth-or-featured-jwst': ['_smooth', '_featured-or-disk', '_star-artifact-zoom'],
    'disk-edge-on-jwst': ['_yes', '_no'],
    'has-spiral-arms-jwst': ['_yes', '_no'],
    'bar-jwst': ['_strong', '_weak', '_no'],
    'bulge-size-jwst': ['_dominant', '_large', '_moderate', '_small', '_none'],
    'how-rounded-jwst': ['_round', '_in-between', '_cigar-shaped'],
    'edge-on-bulge-jwst': ['_boxy', '_none', '_rounded'],
    'spiral-winding-jwst': ['_tight', '_medium', '_loose'],
    'spiral-arm-count-jwst': ['_1', '_2', '_3', '_4', '_more-than-4', '_cant-tell'],
    'clumps-jwst': ['_yes', '_no'],
    'merging-jwst': ['_none', '_minor-disturbance', '_major-disturbance', '_merger'],
    'problem-jwst': ['_star', '_artifact', '_bad-zoom']
}
jwst_ortho_questions, jwst_ortho_label_cols = extract_questions_and_label_cols(jwst_ortho_pairs)

jwst_ortho_dependencies = {
    'smooth-or-featured-jwst': None,  # always asked
    'disk-edge-on-jwst': 'smooth-or-featured-jwst_featured-or-disk',
    'has-spiral-arms-jwst': 'disk-edge-on-jwst_no',
    'bar-jwst': 'disk-edge-on-jwst_no',
    'bulge-size-jwst': 'disk-edge-on-jwst_no',
    'how-rounded-jwst': 'smooth-or-featured-jwst_smooth',
    'edge-on-bulge-jwst': 'disk-edge-on-jwst_yes',
    'spiral-winding-jwst': 'has-spiral-arms-jwst_yes',
    'spiral-arm-count-jwst': 'has-spiral-arms-jwst_yes', # bad naming...
    'merging-jwst': None,  # ignores artifact,
    'clumps-jwst': None,  # ignores artifact
    'problem-jwst': 'smooth-or-featured-jwst_star-artifact-zoom'
}

# JWST COSMOS
jwst_cosmos_ortho_pairs = {
    'smooth-or-featured-jwst': ['_smooth', '_featured-or-disk', '_star-artifact-zoom'],
    'disk-edge-on-jwst': ['_yes', '_no'],
    'has-spiral-arms-jwst': ['_yes', '_no'],
    'bar-jwst': ['_strong', '_weak', '_no'],
    'bulge-size-jwst': ['_dominant', '_large', '_moderate', '_small', '_none'],
    'edge-on-bulge-jwst': ['_boxy', '_none', '_rounded'],
    'spiral-winding-jwst': ['_tight', '_medium', '_loose'],
    'spiral-arm-count-jwst': ['_1', '_2', '_3', '_4', '_more-than-4', '_cant-tell'],
    'clumps-jwst': ['_yes', '_no'],
    'merging-jwst': ['_none', '_minor-disturbance', '_major-disturbance', '_merger'],
    'problem-jwst': ['_star', '_artifact', '_bad-zoom']
}
jwst_cosmos_ortho_questions, jwst_cosmos_ortho_label_cols = extract_questions_and_label_cols(jwst_cosmos_ortho_pairs)

jwst_cosmos_ortho_dependencies = {
    'smooth-or-featured-jwst': None,  # always asked
    'disk-edge-on-jwst': 'smooth-or-featured-jwst_featured-or-disk',
    'has-spiral-arms-jwst': 'disk-edge-on-jwst_no',
    'bar-jwst': 'disk-edge-on-jwst_no',
    'bulge-size-jwst': 'disk-edge-on-jwst_no',
    'edge-on-bulge-jwst': 'disk-edge-on-jwst_yes',
    'spiral-winding-jwst': 'has-spiral-arms-jwst_yes',
    'spiral-arm-count-jwst': 'has-spiral-arms-jwst_yes',
    'merging-jwst': None,  # ignores artifact,
    'clumps-jwst': None,  # ignores artifact
    'problem-jwst': 'smooth-or-featured-jwst_star-artifact-zoom'
}
