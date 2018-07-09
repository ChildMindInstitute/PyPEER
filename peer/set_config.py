def set_parameters():

    if config.json

    eye_mask_path = input()
    train_file = input()
    motion_scrub = input()
    gsr = input()

    configs = {

        "eye_mask_path": eye_mask_path,
        "train_file": train_file,
        "motion_scrub": motion_scrub,
        "global_signal_regression": gsr,

        }

    return configs

if __name__ == "__main__":

    set_parameters()
