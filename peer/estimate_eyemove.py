from peer_func import *

if __name__ == "__main__":

    project_dir, top_data_dir, output_dir, stimulus_path = scaffolding()

    os.chdir(project_dir)

    for i, dataset in enumerate([x for x in os.listdir(top_data_dir) if not x.startswith('.')]):

        data_dir = os.path.abspath(os.path.join(top_data_dir, dataset))

        print(('\nPredicting fixations for participant #{}').format(i+1))
        print('====================================================')

        configs = load_config()

        filepath = os.path.join(data_dir, configs['test_file'])

        print('\nLoad Data')
        print('====================================================')

        data = load_data(filepath)

        if int(configs['use_gsr']):

            print('\nGlobal Signal Regression')
            print('====================================================')

            eye_mask_path = configs['eye_mask_path']
            data = global_signal_regression(data, eye_mask_path)

        raveled_data = [data[:, :, :, vol].ravel() for vol in np.arange(data.shape[3])]

        xmodel, ymodel, xmodel_name, ymodel_name = load_model(output_dir)

        print('\nPredicting Fixations')
        print('====================================================')

        print('Fixations saved to specified output directory.')

        x_fix, y_fix = predict_fixations(xmodel, ymodel, raveled_data)

        x_fixname, y_fixname = save_fixations(x_fix, y_fix, xmodel_name, ymodel_name, output_dir)

        print('\nEstimating Eye Movements')
        print('====================================================')

        estimate_em(x_fix, y_fix, x_fixname, y_fixname, output_dir)

        print('Eye movements saved to specified output directory.')

    print('\n')