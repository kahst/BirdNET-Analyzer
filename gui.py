if __name__ == "__main__":
    import gui.utils as gu
    import gui.single_file as sfa
    import gui.multi_file as mfa
    import gui.train as train
    import gui.segments as gs
    import gui.review as review
    import gui.species as species

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
