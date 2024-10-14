import gui.gui_utils as gu
import gui.gui_single_file as sfa
import gui.gui_multi_file as mfa
import gui.gui_train as train
import gui.gui_segments as gs
import gui.gui_review as review
import gui.gui_species as species

gu.open_window(
    [
        sfa.build_single_analysis_tab,
        mfa.build_multi_analysis_tab,
        train.build_train_tab,
        gs.build_segments_tab,
        review.build_review_tab,
        species.build_species_tab,
    ]
)
