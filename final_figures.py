
# Figure 1 that includes swarm plot in x- and y- directions and example prediction series

def fig_one():

    sns.set()

    def load_data(min_scan=2):

        """Returns list of subjects with at least the specified number of calibration scans

        :param min_scan: Minimum number of scans required to be included in subject list
        :return: Dataframe containing subject IDs, site of MRI collection, number of calibration scans, and motion measures
        :return: List containing subjects with at least min_scan calibration scans
        """

        params = pd.read_csv('model_outputs.csv', index_col='subject', dtype=object)
        params = params.convert_objects(convert_numeric=True)

        if min_scan == 2:

            params = params[(params.scan_count == 2) | (params.scan_count == 3)]

        elif min_scan == 3:

            params = params[params.scan_count == 3]

        sub_list = params.index.values.tolist()

        return params, sub_list

    params, sub_list = load_data(min_scan=2)

    corr_list = []

    for sub in sub_list:

        try:

            temp_df = pd.DataFrame.from_csv \
                ('/data2/Projects/Jake/Human_Brain_Mapping/' + sub + '/gsr0_train1_model_parameters.csv')
            x_corr = temp_df['corr_x'][0]
            y_corr = temp_df['corr_y'][0]

            corr_list.append([sub, x_corr, y_corr])

        except:

            continue

    corr_list.sort(key=lambda x: x[1])

    # Subject with correlation ~ .90: sub-5952373
    # Subject with correlation ~ .75: sub-5852534 for x, sub-5793522 for y
    # Subject with correlation ~ .50: sub-5565631 for x, sub-5601764 for y

    temp_df = pd.DataFrame.from_csv \
        ('/data2/Projects/Jake/Human_Brain_Mapping/' + 'sub-5952373' + '/gsr0_train1_model_calibration_predictions.csv')
    pred_90_x = list(temp_df.x_pred)
    pred_90_y = list(temp_df.y_pred)

    temp_df = pd.DataFrame.from_csv \
        ('/data2/Projects/Jake/Human_Brain_Mapping/' + 'sub-5852534' + '/gsr0_train1_model_calibration_predictions.csv')
    pred_75_x = list(temp_df.x_pred)
    temp_df = pd.DataFrame.from_csv \
        ('/data2/Projects/Jake/Human_Brain_Mapping/' + 'sub-5793522' + '/gsr0_train1_model_calibration_predictions.csv')
    pred_75_y = list(temp_df.y_pred)

    temp_df = pd.DataFrame.from_csv \
        ('/data2/Projects/Jake/Human_Brain_Mapping/' + 'sub-5565631' + '/gsr0_train1_model_calibration_predictions.csv')
    pred_50_x = list(temp_df.x_pred)
    temp_df = pd.DataFrame.from_csv \
        ('/data2/Projects/Jake/Human_Brain_Mapping/' + 'sub-5601764' + '/gsr0_train1_model_calibration_predictions.csv')
    pred_50_y = list(temp_df.y_pred)

    fixations = pd.read_csv('stim_vals.csv')
    x_targets = np.repeat(np.array(fixations['pos_x']), 5* train_sets) * monitor_width / 2
    y_targets = np.repeat(np.array(fixations['pos_y']), 5 * train_sets) * monitor_height / 2

    time_series = range(0, len(x_targets))

    plt.figure(figsize=(10, 5))
    plt.subplot(2, 1, 1)
    plt.ylabel('Horizontal position (px)')
    plt.plot(time_series, x_targets, '.-', color='k', label='Target')
    plt.plot(time_series, pred_90_x, '.-', color='b', label='r ~ .90', alpha=.90)
    plt.plot(time_series, pred_75_x, '.-', color='r', label='r ~ .75', alpha=.75)
    plt.plot(time_series, pred_50_x, '.-', color='g', label='r ~ .50', alpha=.5)
    plt.ylim([-950, 950])
    plt.legend(fancybox=True, prop={'size': 9, 'weight': 'bold'}, bbox_to_anchor=(1, .75))
    plt.subplot(2, 1, 2)
    plt.ylabel('Vertical position (px)')
    plt.xlabel('TR')
    plt.plot(time_series, y_targets, '.-', color='k', label='Target')
    plt.plot(time_series, pred_90_y, '.-', color='b', label='r ~ .90', alpha=.90)
    plt.plot(time_series, pred_75_y, '.-', color='r', label='r ~ .75', alpha=.75)
    plt.plot(time_series, pred_50_y, '.-', color='g', label='r ~ .50', alpha=.5)
    plt.ylim([-775, 775])
    plt.legend(fancybox=True, prop={'size': 9, 'weight': 'bold'}, bbox_to_anchor=(1, .75))
    plt.savefig('/home/json/Desktop/peer_figures_final/demo.png', dpi=600)
    plt.show()

    train_set = '1'

    def create_dict_with_rmse_and_corr_values(sub_list):

        """Creates dictionary that contains list of rmse and corr values for all training combinations

        :return: Dictionary that contains list of rmse and corr values for all training combinations
        """

        file_dict = {'1': '/gsr0_train1_model_parameters.csv',
                     '3': '/gsr0_train3_model_parameters.csv',
                     '13': '/gsr0_train13_model_parameters.csv',
                     '1gsr': '/gsr1_train1_model_parameters.csv'}

        params_dict = {'1': {'corr_x': [], 'corr_y': [], 'rmse_x': [], 'rmse_y': []},
                       '3': {'corr_x': [], 'corr_y': [], 'rmse_x': [], 'rmse_y': []},
                       '13': {'corr_x': [], 'corr_y': [], 'rmse_x': [], 'rmse_y': []},
                       '1gsr': {'corr_x': [], 'corr_y': [], 'rmse_x': [], 'rmse_y': []}}

        for sub in sub_list:

            for train_set in file_dict.keys():

                if os.path.exists(resample_path + sub + file_dict['3']) and (
                        np.isnan(pd.DataFrame.from_csv(resample_path + sub + file_dict['3'])['corr_x'][0]) != True):

                    try:

                        temp_df = pd.DataFrame.from_csv(resample_path + sub + file_dict[train_set])
                        x_corr = temp_df['corr_x'][0]
                        y_corr = temp_df['corr_y'][0]
                        x_rmse = temp_df['rmse_x'][0]
                        y_rmse = temp_df['rmse_y'][0]

                        params_dict[train_set]['corr_x'].append(x_corr)
                        params_dict[train_set]['corr_y'].append(y_corr)
                        params_dict[train_set]['rmse_x'].append(x_rmse)
                        params_dict[train_set]['rmse_y'].append(y_rmse)

                    except:

                        print('Error processing subject ' + sub + ' for ' + train_set)

                else:

                    continue

        return params_dict

    params_dict = create_dict_with_rmse_and_corr_values(sub_list)

    train_name = ['' for x in range(len(params_dict[train_set]['corr_x']))]

    swarm_df = pd.DataFrame({"Pearson's r": params_dict[train_set]['corr_x'],
                             "Pearson's r ": params_dict[train_set]['corr_y'],
                             'rmse_x': params_dict[train_set]['rmse_x'],
                             'rmse_y': params_dict[train_set]['rmse_y'],
                             '': train_name})

    plt.figure(figsize=(5, 5))
    plt.subplot(2, 1, 1)
    ax = sns.swarmplot(x='', y="Pearson's r", data=swarm_df)
    ax.set(title='Correlation Distribution in x')
    plt.ylim([-1, 1.05])
    plt.subplot(2, 1, 2)
    ax = sns.swarmplot(x='', y="Pearson's r ", data=swarm_df)
    ax.set(title='Correlation Distribution in y')
    plt.ylim([-1, 1.05])
    plt.savefig('/home/json/Desktop/peer_figures_final/swarm_gsr0_train1.png', dpi=600)
    plt.show()


def fig_two():

    sns.set()

    def load_data(min_scan=2):

        """Returns list of subjects with at least the specified number of calibration scans

        :param min_scan: Minimum number of scans required to be included in subject list
        :return: Dataframe containing subject IDs, site of MRI collection, number of calibration scans, and motion measures
        :return: List containing subjects with at least min_scan calibration scans
        """

        params = pd.read_csv('model_outputs.csv', index_col='subject', dtype=object)
        params = params.convert_objects(convert_numeric=True)

        if min_scan == 2:

            params = params[(params.scan_count == 2) | (params.scan_count == 3)]

        elif min_scan == 3:

            params = params[params.scan_count == 3]

        sub_list = params.index.values.tolist()

        return params, sub_list

    params, sub_list = load_data(min_scan=2)

    def create_dict_with_rmse_and_corr_values(sub_list):

        """Creates dictionary that contains list of rmse and corr values for all training combinations

        :return: Dictionary that contains list of rmse and corr values for all training combinations
        """

        file_dict = {'1': '/gsr0_train1_model_parameters.csv',
                     '3': '/gsr0_train3_model_parameters.csv',
                     '13': '/gsr0_train13_model_parameters.csv',
                     '1gsr': '/gsr1_train1_model_parameters.csv'}

        params_dict = {'1': {'corr_x': [], 'corr_y': [], 'rmse_x': [], 'rmse_y': []},
                       '3': {'corr_x': [], 'corr_y': [], 'rmse_x': [], 'rmse_y': []},
                       '13': {'corr_x': [], 'corr_y': [], 'rmse_x': [], 'rmse_y': []},
                       '1gsr': {'corr_x': [], 'corr_y': [], 'rmse_x': [], 'rmse_y': []}}

        for sub in sub_list:

            for train_set in file_dict.keys():

                if os.path.exists(resample_path + sub + file_dict['3']) and (
                        np.isnan(pd.DataFrame.from_csv(resample_path + sub + file_dict['3'])['corr_x'][0]) != True):

                    try:

                        temp_df = pd.DataFrame.from_csv(resample_path + sub + file_dict[train_set])
                        x_corr = temp_df['corr_x'][0]
                        y_corr = temp_df['corr_y'][0]
                        x_rmse = temp_df['rmse_x'][0]
                        y_rmse = temp_df['rmse_y'][0]

                        params_dict[train_set]['corr_x'].append(x_corr)
                        params_dict[train_set]['corr_y'].append(y_corr)
                        params_dict[train_set]['rmse_x'].append(x_rmse)
                        params_dict[train_set]['rmse_y'].append(y_rmse)

                    except:

                        print('Error processing subject ' + sub + ' for ' + train_set)

                else:

                    continue

        return params_dict

    params_dict = create_dict_with_rmse_and_corr_values(sub_list)

    #############
    x_ax = '1'
    y_ax = '13'

    val_range = np.linspace(-.6, 1.0, 480)
    slope, intercept, r_val, p_val, std_error = stats.linregress(params_dict[x_ax]['corr_x'], params_dict[y_ax]['corr_x'])
    r2_val = r_val**2
    r2_text = 'r2 value: ' + str(r2_val)[:4]
    r_text = 'r value: ' + str(r_val)[:4]

    plt.figure(figsize=(15, 10))
    plt.subplot(2, 3, 1)
    plt.title('Calibration Scans 1 vs 1&3 in Training')
    plt.xlabel("Pearson's r for Scan 1")
    plt.ylabel("Pearson's r for Scan 1&3")
    plt.xlim([-.6, 1.0])
    plt.ylim([-.6, 1.0])
    plt.scatter(params_dict[x_ax]['corr_x'], params_dict[y_ax]['corr_x'], alpha=.75)
    plt.plot(val_range, slope * (val_range) + intercept, color='r', label=r_text)
    plt.plot([-.6, 1], [-.6, 1], '--', color='k', label='Identical')
    plt.legend(loc=2, prop={'size': 9, 'weight': 'bold'})

    slope, intercept, r_val, p_val, std_error = stats.linregress(params_dict[x_ax]['corr_y'], params_dict[y_ax]['corr_y'])
    r2_val = r_val**2
    r2_text = 'r2 value: ' + str(r2_val)[:4]
    r_text = 'r value: ' + str(r_val)[:4]

    plt.subplot(2, 3, 4)
    plt.xlabel("Pearson's r for Scan 1")
    plt.ylabel("Pearson's r for Scan 1&3")
    plt.xlim([-.6, 1.0])
    plt.ylim([-.6, 1.0])
    plt.scatter(params_dict[x_ax]['corr_y'], params_dict[y_ax]['corr_y'], alpha=.75)
    plt.plot(val_range, slope * (val_range) + intercept, color='r', label=r_text)
    plt.plot([-.6, 1], [-.6, 1], '--', color='k', label='Identical')
    plt.legend(loc=2, prop={'size': 9, 'weight': 'bold'})

    #############
    x_ax = '3'
    y_ax = '13'

    val_range = np.linspace(-.6, 1.0, 480)
    slope, intercept, r_val, p_val, std_error = stats.linregress(params_dict[x_ax]['corr_x'], params_dict[y_ax]['corr_x'])
    r2_val = r_val**2
    r2_text = 'r2 value: ' + str(r2_val)[:4]
    r_text = 'r value: ' + str(r_val)[:4]

    plt.subplot(2, 3, 2)
    plt.title('Calibration Scans 3 vs 1&3 in Training')
    plt.xlabel("Pearson's r for Scan 3")
    plt.ylabel("Pearson's r for Scan 1&3")
    plt.xlim([-.6, 1.0])
    plt.ylim([-.6, 1.0])
    plt.scatter(params_dict[x_ax]['corr_x'], params_dict[y_ax]['corr_x'], alpha=.75)
    plt.plot(val_range, slope * (val_range) + intercept, color='r', label=r_text)
    plt.plot([-.6, 1], [-.6, 1], '--', color='k', label='Identical')
    plt.legend(loc=2, prop={'size': 9, 'weight': 'bold'})

    slope, intercept, r_val, p_val, std_error = stats.linregress(params_dict[x_ax]['corr_y'], params_dict[y_ax]['corr_y'])
    r2_val = r_val**2
    r2_text = 'r2 value: ' + str(r2_val)[:4]
    r_text = 'r value: ' + str(r_val)[:4]

    plt.subplot(2, 3, 5)
    plt.xlabel("Pearson's r for Scan 3")
    plt.ylabel("Pearson's r for Scan 1&3")
    plt.xlim([-.6, 1.0])
    plt.ylim([-.6, 1.0])
    plt.scatter(params_dict[x_ax]['corr_y'], params_dict[y_ax]['corr_y'], alpha=.75)
    plt.plot(val_range, slope * (val_range) + intercept, color='r', label=r_text)
    plt.plot([-.6, 1], [-.6, 1], '--', color='k', label='Identical')
    plt.legend(loc=2, prop={'size': 9, 'weight': 'bold'})

    #############
    x_ax = '1'
    y_ax = '3'

    val_range = np.linspace(-.6, 1.0, 480)
    slope, intercept, r_val, p_val, std_error = stats.linregress(params_dict[x_ax]['corr_x'], params_dict[y_ax]['corr_x'])
    r2_val = r_val**2
    r2_text = 'r2 value: ' + str(r2_val)[:4]
    r_text = 'r value: ' + str(r_val)[:4]

    plt.subplot(2, 3, 3)
    plt.title('Calibration Scans 1 vs 3 in Training')
    plt.xlabel("Pearson's r for Scan 1")
    plt.ylabel("Pearson's r for Scan 3")
    plt.xlim([-.6, 1.0])
    plt.ylim([-.6, 1.0])
    plt.scatter(params_dict[x_ax]['corr_x'], params_dict[y_ax]['corr_x'], alpha=.75)
    plt.plot(val_range, slope * (val_range) + intercept, color='r', label=r_text)
    plt.plot([-.6, 1], [-.6, 1], '--', color='k', label='Identical')
    plt.legend(loc=2, prop={'size': 9, 'weight': 'bold'})

    slope, intercept, r_val, p_val, std_error = stats.linregress(params_dict[x_ax]['corr_y'], params_dict[y_ax]['corr_y'])
    r2_val = r_val**2
    r2_text = 'r2 value: ' + str(r2_val)[:4]
    r_text = 'r value: ' + str(r_val)[:4]

    plt.subplot(2, 3, 6)
    plt.xlabel("Pearson's r for Scan 1")
    plt.ylabel("Pearson's r for Scan 3")
    plt.xlim([-.6, 1.0])
    plt.ylim([-.6, 1.0])
    plt.scatter(params_dict[x_ax]['corr_y'], params_dict[y_ax]['corr_y'], alpha=.75)
    plt.plot(val_range, slope * (val_range) + intercept, color='r', label=r_text)
    plt.plot([-.6, 1], [-.6, 1], '--', color='k', label='Identical')
    plt.legend(loc=2, prop={'size': 9, 'weight': 'bold'})
    plt.savefig('/home/json/Desktop/peer_figures_final/training_comparison.png', dpi=600)
    plt.show()


def fig_three():

    sns.set()

    def load_data(min_scan=2):

        """Returns list of subjects with at least the specified number of calibration scans

        :param min_scan: Minimum number of scans required to be included in subject list
        :return: Dataframe containing subject IDs, site of MRI collection, number of calibration scans, and motion measures
        :return: List containing subjects with at least min_scan calibration scans
        """

        params = pd.read_csv('model_outputs.csv', index_col='subject', dtype=object)
        params = params.convert_objects(convert_numeric=True)

        if min_scan == 2:

            params = params[(params.scan_count == 2) | (params.scan_count == 3)]

        elif min_scan == 3:

            params = params[params.scan_count == 3]

        sub_list = params.index.values.tolist()

        return params, sub_list

    params, sub_list = load_data(min_scan=2)

    def create_dict_with_rmse_and_corr_values(sub_list):

        """Creates dictionary that contains list of rmse and corr values for all training combinations

        :return: Dictionary that contains list of rmse and corr values for all training combinations
        """

        file_dict = {'1': '/gsr0_train1_model_parameters.csv',
                     '3': '/gsr0_train3_model_parameters.csv',
                     '13': '/gsr0_train13_model_parameters.csv',
                     '1gsr': '/gsr1_train1_model_parameters.csv'}

        params_dict = {'1': {'corr_x': [], 'corr_y': [], 'rmse_x': [], 'rmse_y': []},
                       '3': {'corr_x': [], 'corr_y': [], 'rmse_x': [], 'rmse_y': []},
                       '13': {'corr_x': [], 'corr_y': [], 'rmse_x': [], 'rmse_y': []},
                       '1gsr': {'corr_x': [], 'corr_y': [], 'rmse_x': [], 'rmse_y': []}}

        for sub in sub_list:

            for train_set in file_dict.keys():

                if os.path.exists(resample_path + sub + file_dict['3']) and (
                        np.isnan(pd.DataFrame.from_csv(resample_path + sub + file_dict['3'])['corr_x'][0]) != True):

                    try:

                        temp_df = pd.DataFrame.from_csv(resample_path + sub + file_dict[train_set])
                        x_corr = temp_df['corr_x'][0]
                        y_corr = temp_df['corr_y'][0]
                        x_rmse = temp_df['rmse_x'][0]
                        y_rmse = temp_df['rmse_y'][0]

                        params_dict[train_set]['corr_x'].append(x_corr)
                        params_dict[train_set]['corr_y'].append(y_corr)
                        params_dict[train_set]['rmse_x'].append(x_rmse)
                        params_dict[train_set]['rmse_y'].append(y_rmse)

                    except:

                        print('Error processing subject ' + sub + ' for ' + train_set)

                else:

                    continue

        return params_dict

    params_dict = create_dict_with_rmse_and_corr_values(sub_list)

    x_ax = '1'
    y_ax = '1gsr'

    val_range = np.linspace(-.6, 1.0, 480)
    slope, intercept, r_val, p_val, std_error = stats.linregress(params_dict[x_ax]['corr_x'], params_dict[y_ax]['corr_x'])
    r2_val = r_val**2
    r2_text = 'r2 value: ' + str(r2_val)[:4]
    r_text = 'r value: ' + str(r_val)[:4]

    plt.figure(figsize=(10, 5))
    plt.subplot(121)
    plt.title('GSR Comparison in x')
    plt.xlabel("Pearson's r for Scan 1")
    plt.ylabel("Pearson's r for Scan 1 with GSR")
    plt.xlim([-.6, 1.0])
    plt.ylim([-.6, 1.0])
    plt.scatter(params_dict[x_ax]['corr_x'], params_dict[y_ax]['corr_x'], alpha=.75)
    plt.plot(val_range, slope * (val_range) + intercept, color='r', label=r_text)
    plt.plot([-.6, 1], [-.6, 1], '--', color='k', label='Identical')
    plt.legend(loc=2, prop={'size': 9, 'weight': 'bold'})

    slope, intercept, r_val, p_val, std_error = stats.linregress(params_dict[x_ax]['corr_y'], params_dict[y_ax]['corr_y'])
    r2_val = r_val**2
    r2_text = 'r2 value: ' + str(r2_val)[:4]
    r_text = 'r value: ' + str(r_val)[:4]

    plt.subplot(122)
    plt.title('GSR Comparison in y')
    plt.xlabel("Pearson's r for Scan 1")
    plt.ylabel("Pearson's r for Scan 1 with GSR")
    plt.xlim([-.6, 1.0])
    plt.ylim([-.6, 1.0])
    plt.scatter(params_dict[x_ax]['corr_y'], params_dict[y_ax]['corr_y'], alpha=.75)
    plt.plot(val_range, slope * (val_range) + intercept, color='r', label=r_text)
    plt.plot([-.6, 1], [-.6, 1], '--', color='k', label='Identical')
    plt.legend(loc=2, prop={'size': 9, 'weight': 'bold'})
    plt.savefig('/home/json/Desktop/peer_figures_final/gsr_comparison.png', dpi=600)
    plt.show()


def fig_four():

    viewtype = 'calibration'
    modality = 'peer'
    cscheme = 'inferno'

    monitor_width = 1680
    monitor_height = 1050


    def load_data(min_scan=2):

        """Returns list of subjects with at least the specified number of calibration scans

        :param min_scan: Minimum number of scans required to be included in subject list
        :return: Dataframe containing subject IDs, site of MRI collection, number of calibration scans, and motion measures
        :return: List containing subjects with at least min_scan calibration scans
        """

        params = pd.read_csv('model_outputs.csv', index_col='subject', dtype=object)
        params = params.convert_objects(convert_numeric=True)

        if min_scan == 2:

            params = params[(params.scan_count == 2) | (params.scan_count == 3)]

        elif min_scan == 3:

            params = params[params.scan_count == 3]

        sub_list = params.index.values.tolist()

        return params, sub_list

    params, sub_list = load_data(min_scan=2)

    x_stack = []
    y_stack = []
    sorted_by = 'mean_fd'

    params = params.sort_values(by=[sorted_by])
    sub_list = params.index.values.tolist()

    filename_dict = {'calibration': {'name': '/gsr0_train1_model_calibration_predictions.csv', 'num_vol': 135},
                     'tp': {'name': '/gsr0_train1_model_tp_predictions.csv', 'num_vol': 250},
                     'dm': {'name': '/gsr0_train1_model_dm_predictions.csv', 'num_vol': 750}}

    fixations = pd.read_csv('stim_vals.csv')
    x_targets = np.tile(np.repeat(np.array(fixations['pos_x']), 5) * monitor_width / 2, 1)
    y_targets = np.tile(np.repeat(np.array(fixations['pos_y']), 5) * monitor_height / 2, 1)

    for sub in sub_list:

        try:

            if modality == 'peer':

                temp_df = pd.DataFrame.from_csv(resample_path + sub + filename_dict[viewtype]['name'])

            elif modality == 'et':

                temp_df = pd.DataFrame.from_csv('/data2/Projects/Jake/Human_Brain_Mapping/' + sub + '/et_device_pred.csv')

            x_series = list(temp_df['x_pred'])
            y_series = list(temp_df['y_pred'])

            if (len(x_series) == filename_dict[viewtype]['num_vol']) and (len(y_series) == filename_dict[viewtype]['num_vol']):

                x_series = [x if abs(x) < monitor_width/2 + .1*monitor_width else 0 for x in x_series]
                y_series = [x if abs(x) < monitor_height/2 + .1*monitor_height else 0 for x in y_series]

                x_stack.append(x_series)
                y_stack.append(y_series)

        except:

            print('Error processing subject ' + sub)

    arr = np.zeros(len(x_series))
    arrx = np.array([-np.round(monitor_width / 2, 0) for x in arr])
    arry = np.array([-np.round(monitor_height / 2, 0) for x in arr])

    if viewtype == 'calibration':

        for num in range(int(np.round(len(sub_list) * .02, 0))):
            x_stack.append(arrx)
            y_stack.append(arry)

        for num in range(int(np.round(len(sub_list) * .02, 0))):
            x_stack.append(x_targets)
            y_stack.append(y_targets)

    else:

        avg_series_x = np.mean(x_stack, axis=0)
        avg_series_y = np.mean(y_stack, axis=0)

        for num in range(int(np.round(len(sub_list) * .02, 0))):
            x_stack.append(arrx)
            y_stack.append(arry)

        for num in range(int(np.round(len(sub_list) * .02, 0))):
            x_stack.append(avg_series_x)
            y_stack.append(avg_series_y)

    x_hm = np.stack(x_stack)
    y_hm = np.stack(y_stack)

    x_spacing = len(x_hm[0])

    fig = plt.figure(figsize=(10, 10))
    fig.suptitle('Calibration Scan Heatmaps')
    plt.subplot(221)
    plt.title('Ranked by MeanFD')
    ax = sns.heatmap(x_hm, cmap=cscheme)
    ax.set(ylabel='Subjects')
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))
    ax.xaxis.set_major_locator(ticker.MultipleLocator(base=np.round(x_spacing/5, 0)))
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(base=100))

    plt.subplot(223)
    ax = sns.heatmap(y_hm, cmap=cscheme)
    ax.set(ylabel='Subjects')
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))
    ax.xaxis.set_major_locator(ticker.MultipleLocator(base=np.round(x_spacing/5, 0)))
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(base=100))

    #################

    x_stack = []
    y_stack = []
    sorted_by = 'dvars'

    params = params.sort_values(by=[sorted_by], ascending=False)
    sub_list = params.index.values.tolist()

    filename_dict = {'calibration': {'name': '/gsr0_train1_model_calibration_predictions.csv', 'num_vol': 135},
                     'tp': {'name': '/gsr0_train1_model_tp_predictions.csv', 'num_vol': 250},
                     'dm': {'name': '/gsr0_train1_model_dm_predictions.csv', 'num_vol': 750}}

    fixations = pd.read_csv('stim_vals.csv')
    x_targets = np.tile(np.repeat(np.array(fixations['pos_x']), 5) * monitor_width / 2, 1)
    y_targets = np.tile(np.repeat(np.array(fixations['pos_y']), 5) * monitor_height / 2, 1)

    for sub in sub_list:

        try:

            if modality == 'peer':

                temp_df = pd.DataFrame.from_csv(resample_path + sub + filename_dict[viewtype]['name'])

            elif modality == 'et':

                temp_df = pd.DataFrame.from_csv('/data2/Projects/Jake/Human_Brain_Mapping/' + sub + '/et_device_pred.csv')

            x_series = list(temp_df['x_pred'])
            y_series = list(temp_df['y_pred'])

            if (len(x_series) == filename_dict[viewtype]['num_vol']) and (len(y_series) == filename_dict[viewtype]['num_vol']):

                x_series = [x if abs(x) < monitor_width/2 + .1*monitor_width else 0 for x in x_series]
                y_series = [x if abs(x) < monitor_height/2 + .1*monitor_height else 0 for x in y_series]

                x_stack.append(x_series)
                y_stack.append(y_series)

        except:

            print('Error processing subject ' + sub)

    arr = np.zeros(len(x_series))
    arrx = np.array([-np.round(monitor_width / 2, 0) for x in arr])
    arry = np.array([-np.round(monitor_height / 2, 0) for x in arr])

    if viewtype == 'calibration':

        for num in range(int(np.round(len(sub_list) * .02, 0))):
            x_stack.append(arrx)
            y_stack.append(arry)

        for num in range(int(np.round(len(sub_list) * .02, 0))):
            x_stack.append(x_targets)
            y_stack.append(y_targets)

    else:

        avg_series_x = np.mean(x_stack, axis=0)
        avg_series_y = np.mean(y_stack, axis=0)

        for num in range(int(np.round(len(sub_list) * .02, 0))):
            x_stack.append(arrx)
            y_stack.append(arry)

        for num in range(int(np.round(len(sub_list) * .02, 0))):
            x_stack.append(avg_series_x)
            y_stack.append(avg_series_y)

    x_hm = np.stack(x_stack)
    y_hm = np.stack(y_stack)

    x_spacing = len(x_hm[0])

    plt.subplot(222)
    plt.title('Ranked by DVARS')
    ax = sns.heatmap(x_hm, cmap=cscheme)
    ax.set(xlabel='Volumes', ylabel='Subjects')
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))
    ax.xaxis.set_major_locator(ticker.MultipleLocator(base=np.round(x_spacing/5, 0)))
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(base=100))

    plt.subplot(224)
    ax = sns.heatmap(y_hm, cmap=cscheme)
    ax.set(xlabel='Volumes', ylabel='Subjects')
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))
    ax.xaxis.set_major_locator(ticker.MultipleLocator(base=np.round(x_spacing/5, 0)))
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(base=100))
    plt.savefig('/home/json/Desktop/peer_figures_final/calibration_heatmap.png', dpi=600)
    plt.show()


def fig_five():

    # To compare linear relationship between Pearson's r and factors that may affect head motion

    def load_data(min_scan=2):

        """Returns list of subjects with at least the specified number of calibration scans

        :param min_scan: Minimum number of scans required to be included in subject list
        :return: Dataframe containing subject IDs, site of MRI collection, number of calibration scans, and motion measures
        :return: List containing subjects with at least min_scan calibration scans
        """

        params = pd.read_csv('model_outputs.csv', index_col='subject', dtype=object)
        params = params.convert_objects(convert_numeric=True)

        if min_scan == 2:

            params = params[(params.scan_count == 2) | (params.scan_count == 3)]

        elif min_scan == 3:

            params = params[params.scan_count == 3]

        sub_list = params.index.values.tolist()

        return params, sub_list

    params, sub_list = load_data(min_scan=2)

    motion_dict = {'mean_fd': [], 'dvars': [], 'corr_x': [], 'corr_y': [], 'age': [], 'fsiq': []}
    temp_df = pd.DataFrame.from_csv('/home/json/Desktop/peer/Peer_pheno.csv')
    temp_df = temp_df.drop_duplicates()

    for sub in sub_list:

        try:

            mean_fd = params.loc[sub, 'mean_fd']
            dvars = params.loc[sub, 'dvars']
            corr_x = pd.DataFrame.from_csv(resample_path + sub + '//gsr0_train1_model_parameters.csv')['corr_x'][0]
            corr_y = pd.DataFrame.from_csv(resample_path + sub + '//gsr0_train1_model_parameters.csv')['corr_y'][0]

            age_val = temp_df.loc[int(sub.strip('sub-')), 'Age']
            fsiq_val = temp_df.loc[int(sub.strip('sub-')), 'FSIQ']

            motion_dict['mean_fd'].append(mean_fd)
            motion_dict['dvars'].append(dvars)
            motion_dict['corr_x'].append(corr_x)
            motion_dict['corr_y'].append(corr_y)
            motion_dict['age'].append(age_val)
            motion_dict['fsiq'].append(fsiq_val)

        except:

            continue

    mean_fd_list = []
    dvars_list = []
    corr_x_list = []
    corr_y_list = []
    age_list = []
    fsiq_list = []

    motion_type = 'mean_fd'
    thresh = 100000

    for num in range(len(motion_dict[motion_type])):

        if motion_dict[motion_type][num] < thresh:

            mean_fd_list.append(motion_dict['mean_fd'][num])
            dvars_list.append(motion_dict['dvars'][num])
            corr_x_list.append(motion_dict['corr_x'][num])
            corr_y_list.append(motion_dict['corr_y'][num])
            age_list.append(motion_dict['age'][num])
            fsiq_list.append(motion_dict['fsiq'][num])

    motion_dict['mean_fd'] = mean_fd_list
    motion_dict['dvars'] = dvars_list
    motion_dict['corr_x'] = corr_x_list
    motion_dict['corr_y'] = corr_y_list
    motion_dict['age'] = age_list
    motion_dict['fsiq'] = fsiq_list

    ######## MeanFD and DVARS

    motion_type = 'mean_fd'

    val_range = np.linspace(np.nanmin(motion_dict[motion_type]), np.nanmax(motion_dict[motion_type]),
                            len(motion_dict[motion_type]))

    slope, intercept, r_val, p_val, std_error = stats.linregress(motion_dict[motion_type], motion_dict['corr_x'])
    r2_val = r_val**2
    r2_text = 'r2 value: ' + str(r2_val)[:5]
    r_text = 'r value: ' + str(r_val)[:5]

    plt.figure(figsize=(10, 10))
    plt.subplot(221)
    plt.title('MeanFD Effects on Model Accuracy')
    plt.xlabel('MeanFD')
    plt.ylabel("Pearson's r")
    plt.ylim([-.6, 1.0])
    plt.scatter(motion_dict[motion_type], motion_dict['corr_x'], alpha=.75)
    plt.plot(val_range, slope * (val_range) + intercept, color='r', label=r_text)
    plt.legend(loc=4, prop={'weight': 'bold', 'size': 9})

    slope, intercept, r_val, p_val, std_error = stats.linregress(motion_dict[motion_type], motion_dict['corr_y'])
    r2_val = r_val**2
    r2_text = 'r2 value: ' + str(r2_val)[:5]
    r_text = 'r value: ' + str(r_val)[:5]

    plt.subplot(223)
    plt.xlabel('MeanFD')
    plt.ylabel("Pearson's r")
    plt.ylim([-.6, 1.0])
    plt.scatter(motion_dict[motion_type], motion_dict['corr_y'], alpha=.75)
    plt.plot(val_range, slope * (val_range) + intercept, color='r', label=r_text)
    plt.legend(loc=4, prop={'weight': 'bold', 'size': 9})

    ########

    motion_type = 'dvars'

    val_range = np.linspace(np.nanmin(motion_dict[motion_type]), np.nanmax(motion_dict[motion_type]),
                            len(motion_dict[motion_type]))

    slope, intercept, r_val, p_val, std_error = stats.linregress(motion_dict[motion_type], motion_dict['corr_x'])
    r2_val = r_val**2
    r2_text = 'r2 value: ' + str(r2_val)[:5]
    r_text = 'r value: ' + str(r_val)[:5]

    plt.subplot(222)
    plt.title('DVARS Effects on Model Accuracy')
    plt.xlabel('DVARS')
    plt.ylabel("Pearson's r")
    plt.ylim([-.6, 1.0])
    plt.scatter(motion_dict[motion_type], motion_dict['corr_x'], alpha=.75)
    plt.plot(val_range, slope * (val_range) + intercept, color='r', label=r_text)
    plt.legend(loc=4, prop={'weight': 'bold', 'size': 9})

    slope, intercept, r_val, p_val, std_error = stats.linregress(motion_dict[motion_type], motion_dict['corr_y'])
    r2_val = r_val**2
    r2_text = 'r2 value: ' + str(r2_val)[:5]
    r_text = 'r value: ' + str(r_val)[:5]

    plt.subplot(224)
    plt.xlabel('DVARS')
    plt.ylabel("Pearson's r")
    plt.ylim([-.6, 1.0])
    plt.scatter(motion_dict[motion_type], motion_dict['corr_y'], alpha=.75)
    plt.plot(val_range, slope * (val_range) + intercept, color='r', label=r_text)
    plt.legend(loc=4, prop={'weight': 'bold', 'size': 9})
    plt.savefig('/home/json/Desktop/peer_figures_final/motion_effects.png', dpi=600)
    plt.show()

    ######## Age and FSIQ

    motion_type = 'age'

    val_range = np.linspace(np.nanmin(motion_dict[motion_type]), np.nanmax(motion_dict[motion_type]),
                            len(motion_dict[motion_type]))

    slope, intercept, r_val, p_val, std_error = stats.linregress(motion_dict[motion_type], motion_dict['corr_x'])
    r2_val = r_val**2
    r2_text = 'r2 value: ' + str(r2_val)[:5]
    r_text = 'r value: ' + str(r_val)[:5]

    plt.figure(figsize=(10, 10))
    plt.subplot(221)
    plt.title('Age Effects on Model Accuracy')
    plt.xlabel('Age')
    plt.ylabel("Pearson's r")
    plt.ylim([-.6, 1.0])
    plt.scatter(motion_dict[motion_type], motion_dict['corr_x'], alpha=.75)
    plt.plot(val_range, slope * (val_range) + intercept, color='r', label=r_text)
    plt.legend(loc=4, prop={'weight': 'bold', 'size': 9})

    slope, intercept, r_val, p_val, std_error = stats.linregress(motion_dict[motion_type], motion_dict['corr_y'])
    r2_val = r_val**2
    r2_text = 'r2 value: ' + str(r2_val)[:5]
    r_text = 'r value: ' + str(r_val)[:5]

    plt.subplot(223)
    plt.xlabel('Age')
    plt.ylabel("Pearson's r")
    plt.ylim([-.6, 1.0])
    plt.scatter(motion_dict[motion_type], motion_dict['corr_y'], alpha=.75)
    plt.plot(val_range, slope * (val_range) + intercept, color='r', label=r_text)
    plt.legend(loc=4, prop={'weight': 'bold', 'size': 9})

    ########

    motion_type = 'fsiq'

    fsiq_list = []
    corr_x_list = []
    corr_y_list = []

    for item in range(len(motion_dict['fsiq'])):

        if ~np.isnan(motion_dict['fsiq'][item]):
            fsiq_list.append(motion_dict['fsiq'][item])
            corr_x_list.append(motion_dict['corr_x'][item])
            corr_y_list.append(motion_dict['corr_y'][item])

    val_range = np.linspace(np.nanmin(motion_dict[motion_type]), np.nanmax(motion_dict[motion_type]),
                            len(motion_dict[motion_type]))

    slope, intercept, r_val, p_val, std_error = stats.linregress(fsiq_list, corr_x_list)
    r2_val = r_val**2
    r2_text = 'r2 value: ' + str(r2_val)[:5]
    r_text = 'r value: ' + str(r_val)[:5]

    plt.subplot(222)
    plt.title('FSIQ Effects on Model Accuracy')
    plt.xlabel('FSIQ')
    plt.ylabel("Pearson's r")
    plt.ylim([-.6, 1.0])
    plt.scatter(motion_dict[motion_type], motion_dict['corr_x'], alpha=.75)
    plt.plot(val_range, slope * (val_range) + intercept, color='r', label=r_text)
    plt.legend(loc=4, prop={'weight': 'bold', 'size': 9})

    slope, intercept, r_val, p_val, std_error = stats.linregress(fsiq_list, corr_y_list)
    r2_val = r_val**2
    r2_text = 'r2 value: ' + str(r2_val)[:5]
    r_text = 'r value: ' + str(r_val)[:5]

    plt.subplot(224)
    plt.xlabel('FSIQ')
    plt.ylabel("Pearson's r")
    plt.ylim([-.6, 1.0])
    plt.scatter(motion_dict[motion_type], motion_dict['corr_y'], alpha=.75)
    plt.plot(val_range, slope * (val_range) + intercept, color='r', label=r_text)
    plt.legend(loc=4, prop={'weight': 'bold', 'size': 9})
    plt.savefig('/home/json/Desktop/peer_figures_final/age_fsiq_effects.png', dpi=600)
    plt.show()


def fig_six():

    viewtype = 'dm'
    modality = 'peer'
    cscheme = 'inferno'

    def load_data(min_scan=2):

        """Returns list of subjects with at least the specified number of calibration scans

        :param min_scan: Minimum number of scans required to be included in subject list
        :return: Dataframe containing subject IDs, site of MRI collection, number of calibration scans, and motion measures
        :return: List containing subjects with at least min_scan calibration scans
        """

        params = pd.read_csv('model_outputs.csv', index_col='subject', dtype=object)
        params = params.convert_objects(convert_numeric=True)

        if min_scan == 2:

            params = params[(params.scan_count == 2) | (params.scan_count == 3)]

        elif min_scan == 3:

            params = params[params.scan_count == 3]

        sub_list = params.index.values.tolist()

        return params, sub_list

    params, sub_list = load_data(min_scan=2)

    x_stack = []
    y_stack = []
    sorted_by = 'mean_fd'

    params = params.sort_values(by=[sorted_by])
    sub_list = params.index.values.tolist()

    filename_dict = {'calibration': {'name': '/gsr0_train1_model_calibration_predictions.csv', 'num_vol': 135},
                     'tp': {'name': '/gsr0_train1_model_tp_predictions.csv', 'num_vol': 250},
                     'dm': {'name': '/gsr0_train1_model_dm_predictions.csv', 'num_vol': 750}}

    fixations = pd.read_csv('stim_vals.csv')
    x_targets = np.tile(np.repeat(np.array(fixations['pos_x']), 5) * monitor_width / 2, 1)
    y_targets = np.tile(np.repeat(np.array(fixations['pos_y']), 5) * monitor_height / 2, 1)

    for sub in sub_list:

        try:

            if modality == 'peer':

                temp_df = pd.DataFrame.from_csv(resample_path + sub + filename_dict[viewtype]['name'])

            elif modality == 'et':

                temp_df = pd.DataFrame.from_csv('/data2/Projects/Jake/Human_Brain_Mapping/' + sub + '/et_device_pred.csv')

            x_series = list(temp_df['x_pred'])
            y_series = list(temp_df['y_pred'])

            if (len(x_series) == filename_dict[viewtype]['num_vol']) and (len(y_series) == filename_dict[viewtype]['num_vol']):

                x_series = [x if abs(x) < monitor_width/2 + .1*monitor_width else 0 for x in x_series]
                y_series = [x if abs(x) < monitor_height/2 + .1*monitor_height else 0 for x in y_series]

                x_stack.append(x_series)
                y_stack.append(y_series)

        except:

            print('Error processing subject ' + sub)

    arr = np.zeros(len(x_series))
    arrx = np.array([-np.round(monitor_width / 2, 0) for x in arr])
    arry = np.array([-np.round(monitor_height / 2, 0) for x in arr])

    if viewtype == 'calibration':

        for num in range(int(np.round(len(sub_list) * .02, 0))):
            x_stack.append(arrx)
            y_stack.append(arry)

        for num in range(int(np.round(len(sub_list) * .02, 0))):
            x_stack.append(x_targets)
            y_stack.append(y_targets)

    else:

        avg_series_x = np.mean(x_stack, axis=0)
        avg_series_y = np.mean(y_stack, axis=0)

        for num in range(int(np.round(len(sub_list) * .02, 0))):
            x_stack.append(arrx)
            y_stack.append(arry)

        for num in range(int(np.round(len(sub_list) * .02, 0))):
            x_stack.append(avg_series_x)
            y_stack.append(avg_series_y)

    x_hm = np.stack(x_stack)
    y_hm = np.stack(y_stack)

    x_spacing = len(x_hm[0])

    fig = plt.figure(figsize=(10, 10))
    fig.suptitle('Despicable Me Heatmaps')
    plt.subplot(221)
    plt.title('Ranked by MeanFD')
    ax = sns.heatmap(x_hm, cmap=cscheme)
    ax.set(xlabel='TR', ylabel='Subjects')
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))
    ax.xaxis.set_major_locator(ticker.MultipleLocator(base=np.round(x_spacing/5, 0)))
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(base=100))

    plt.subplot(223)
    ax = sns.heatmap(y_hm, cmap=cscheme)
    ax.set(xlabel='TR', ylabel='Subjects')
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))
    ax.xaxis.set_major_locator(ticker.MultipleLocator(base=np.round(x_spacing/5, 0)))
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(base=100))

    #################

    x_stack = []
    y_stack = []
    sorted_by = 'dvars'

    params = params.sort_values(by=[sorted_by], ascending=False)
    sub_list = params.index.values.tolist()

    filename_dict = {'calibration': {'name': '/gsr0_train1_model_calibration_predictions.csv', 'num_vol': 135},
                     'tp': {'name': '/gsr0_train1_model_tp_predictions.csv', 'num_vol': 250},
                     'dm': {'name': '/gsr0_train1_model_dm_predictions.csv', 'num_vol': 750}}

    fixations = pd.read_csv('stim_vals.csv')
    x_targets = np.tile(np.repeat(np.array(fixations['pos_x']), 5) * monitor_width / 2, 1)
    y_targets = np.tile(np.repeat(np.array(fixations['pos_y']), 5) * monitor_height / 2, 1)

    for sub in sub_list:

        try:

            if modality == 'peer':

                temp_df = pd.DataFrame.from_csv(resample_path + sub + filename_dict[viewtype]['name'])

            elif modality == 'et':

                temp_df = pd.DataFrame.from_csv('/data2/Projects/Jake/Human_Brain_Mapping/' + sub + '/et_device_pred.csv')

            x_series = list(temp_df['x_pred'])
            y_series = list(temp_df['y_pred'])

            if (len(x_series) == filename_dict[viewtype]['num_vol']) and (len(y_series) == filename_dict[viewtype]['num_vol']):

                x_series = [x if abs(x) < monitor_width/2 + .1*monitor_width else 0 for x in x_series]
                y_series = [x if abs(x) < monitor_height/2 + .1*monitor_height else 0 for x in y_series]

                x_stack.append(x_series)
                y_stack.append(y_series)

        except:

            print('Error processing subject ' + sub)

    arr = np.zeros(filename_dict[viewtype]['num_vol'])
    arrx = np.array([-np.round(monitor_width / 2, 0) for x in arr])
    arry = np.array([-np.round(monitor_height / 2, 0) for x in arr])

    if viewtype == 'calibration':

        for num in range(int(np.round(len(sub_list) * .02, 0))):
            x_stack.append(arrx)
            y_stack.append(arry)

        for num in range(int(np.round(len(sub_list) * .02, 0))):
            x_stack.append(x_targets)
            y_stack.append(y_targets)

    else:

        avg_series_x = np.mean(x_stack, axis=0)
        avg_series_y = np.mean(y_stack, axis=0)

        for num in range(int(np.round(len(sub_list) * .02, 0))):
            x_stack.append(arrx)
            y_stack.append(arry)

        for num in range(int(np.round(len(sub_list) * .02, 0))):
            x_stack.append(avg_series_x)
            y_stack.append(avg_series_y)

    x_hm = np.stack(x_stack)
    y_hm = np.stack(y_stack)

    x_spacing = len(x_hm[0])

    plt.subplot(222)
    plt.title('Ranked by DVARS')
    ax = sns.heatmap(x_hm, cmap=cscheme)
    ax.set(xlabel='TR', ylabel='Subjects')
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))
    ax.xaxis.set_major_locator(ticker.MultipleLocator(base=np.round(x_spacing/5, 0)))
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(base=100))

    plt.subplot(224)
    ax = sns.heatmap(y_hm, cmap=cscheme)
    ax.set(xlabel='TR', ylabel='Subjects')
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))
    ax.xaxis.set_major_locator(ticker.MultipleLocator(base=np.round(x_spacing/5, 0)))
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(base=100))
    plt.savefig('/home/json/Desktop/peer_figures_final/' + viewtype + '_heatmap.png', dpi=600)
    plt.show()


def fig_seven():

    eye_mask = nib.load('/data2/Projects/Jake/eye_masks/2mm_eye_corrected.nii.gz')
    resample_path = '/data2/Projects/Jake/Human_Brain_Mapping/'

    bad_subs_list_tp = []
    bad_subs_list_dm = []

    with open('/data2/Projects/Lei/Peers/scripts/data_check/TP_bad_sub.txt') as f:
        bad_subs_list_tp = f.readlines()
    with open('/data2/Projects/Lei/Peers/scripts/data_check/DM_bad_sub.txt') as f:
        bad_subs_list_dm = f.readlines()

    bad_subs_list_tp = [x.strip('\n') for x in bad_subs_list_tp]
    bad_subs_list_dm = [x.strip('\n') for x in bad_subs_list_dm]

    def load_data(min_scan=2):

        """Returns list of subjects with at least the specified number of calibration scans

        :param min_scan: Minimum number of scans required to be included in subject list
        :return: Dataframe containing subject IDs, site of MRI collection, number of calibration scans, and motion measures
        :return: List containing subjects with at least min_scan calibration scans
        """

        params = pd.read_csv('model_outputs.csv', index_col='subject', dtype=object)
        params = params.convert_objects(convert_numeric=True)

        if min_scan == 2:

            params = params[(params.scan_count == 2) | (params.scan_count == 3)]

        elif min_scan == 3:

            params = params[params.scan_count == 3]

        sub_list = params.index.values.tolist()

        return params, sub_list

    params, sub_list = load_data(min_scan=2)

    expected_value = 250

    pd_dict_x = {}
    pd_dict_y = {}

    corr_matrix_tp_x = []
    corr_matrix_tp_y = []

    for sub in sub_list:

        if sub not in bad_subs_list_tp:

            try:

                tp_x = np.array(pd.read_csv(resample_path + sub + '/gsr0_train1_model_tp_predictions.csv')['x_pred'])
                tp_y = np.array(pd.read_csv(resample_path + sub + '/gsr0_train1_model_tp_predictions.csv')['y_pred'])

                if len(tp_x) == expected_value:
                    corr_matrix_tp_x.append(tp_x)
                    corr_matrix_tp_y.append(tp_y)

                    pd_dict_x[str('tp' + sub)] = tp_x
                    pd_dict_y[str('tp' + sub)] = tp_y

            except:

                continue

    corr_matrix_dm_x = []
    corr_matrix_dm_y = []

    for sub in sub_list:

        if sub not in bad_subs_list_dm:

            try:

                dm_x = np.array(pd.read_csv(resample_path + sub + '/gsr0_train1_model_dm_predictions.csv')['x_pred'][:250])
                dm_y = np.array(pd.read_csv(resample_path + sub + '/gsr0_train1_model_dm_predictions.csv')['y_pred'][:250])

                if len(dm_x) == expected_value:
                    corr_matrix_dm_x.append(dm_x)
                    corr_matrix_dm_y.append(dm_y)

                    pd_dict_x[str('dm' + sub)] = dm_x
                    pd_dict_y[str('dm' + sub)] = dm_y

            except:

                continue

    pd_dict_x['index'] = range(len(pd_dict_x[str('dm' + sub)]))
    pd_dict_y['index'] = range(len(pd_dict_y[str('dm' + sub)]))

    df_x = pd.DataFrame.from_dict(pd_dict_x)
    df_x = df_x.set_index('index')
    df_x = df_x.reindex_axis(sorted(df_x.columns), axis=1)
    df_y = pd.DataFrame.from_dict(pd_dict_y)
    df_y = df_y.set_index('index')
    df_y = df_y.reindex_axis(sorted(df_y.columns), axis=1)

    corr_x = df_x.corr(method='pearson')
    corr_y = df_y.corr(method='pearson')

    # OPTIONAL - INCLUDE ONLY BELOW DIAGNOAL
    mask_x = np.zeros_like(corr_x)
    mask_x[np.triu_indices_from(mask_x)] = True
    mask_y = np.zeros_like(corr_y)
    mask_y[np.triu_indices_from(mask_y)] = True

    within_x = []
    between_x = []

    for item1 in corr_x.columns:
        for item2 in corr_x.columns:
            if (item1[:2] == item2[:2]) and (item1 != item2):
                within_x.append(corr_x[item1][item2])
            elif (item1[:2] != item2[:2]) and (item1 != item2):
                between_x.append(corr_x[item1][item2])

    print('Completed x matrix')

    final_dict_x = {}

    final_dict_x["Pearson's r"] = within_x + between_x
    final_dict_x[''] = ['Within Movie']*len(within_x) + ['Between Movie']*len(between_x)

    final_df_x = pd.DataFrame.from_dict(final_dict_x)

    within_y = []
    between_y = []

    for item1 in corr_y.columns:
        for item2 in corr_y.columns:
            if (item1[:2] == item2[:2]) and (item1 != item2):
                within_y.append(corr_y[item1][item2])
            elif (item1[:2] != item2[:2]) and (item1 != item2):
                between_y.append(corr_y[item1][item2])

    print('Completed y matrix')

    final_dict_y = {}

    final_dict_y["Pearson's r"] = within_y + between_y
    final_dict_y[''] = ['Within Movie']*len(within_y) + ['Between Movie']*len(between_y)

    final_df_y = pd.DataFrame.from_dict(final_dict_y)

    plt.figure(figsize=(10, 10))
    plt.subplot(221)
    plt.title('Correlation Matrices for DM and TP')
    sns.heatmap(corr_x, cmap='inferno', xticklabels=False, yticklabels=False)
    plt.xticks([213, 607], ['DM', 'TP'])
    plt.yticks([213, 607], ['DM', 'TP'])

    plt.subplot(223)
    sns.heatmap(corr_y, cmap='inferno', xticklabels=False, yticklabels=False)
    plt.xticks([213, 607], ['DM', 'TP'])
    plt.yticks([213, 607], ['DM', 'TP'])

    plt.subplot(222)
    plt.title('Within and Between Movie Correlations')
    # ax = sns.boxplot(x='', y="Pearson's r", data=final_df_x)
    ax = sns.violinplot(x='', y="Pearson's r", data=final_df_x, scale='count')
    # ax = sns.stripplot(x='', y="Pearson's r", data=final_df_x, jitter=True)
    plt.subplot(224)
    ax = sns.violinplot(x='', y="Pearson's r", data=final_df_y, scale='count')
    plt.savefig('/home/json/Desktop/peer_figures_final/movieprint.png', dpi=600)
    plt.show()


def fig_eight():

    tt_split=.5

    def load_data(min_scan=2):

        """Returns list of subjects with at least the specified number of calibration scans

        :param min_scan: Minimum number of scans required to be included in subject list
        :return: Dataframe containing subject IDs, site of MRI collection, number of calibration scans, and motion measures
        :return: List containing subjects with at least min_scan calibration scans
        """

        params = pd.read_csv('model_outputs.csv', index_col='subject', dtype=object)
        params = params.convert_objects(convert_numeric=True)

        if min_scan == 2:

            params = params[(params.scan_count == 2) | (params.scan_count == 3)]

        elif min_scan == 3:

            params = params[params.scan_count == 3]

        sub_list = params.index.values.tolist()

        return params, sub_list

    params, sub_list = load_data(min_scan=2)

    def create_corr_matrix(sub_list):

        corr_matrix_tp_x = []
        corr_matrix_dm_x = []
        corr_matrix_tp_y = []
        corr_matrix_dm_y = []
        count = 0

        for sub in sub_list:

            try:

                if count == 0:
                    '/data2/Projects/Jake/Human_Brain_Mapping/sub-5002891/gsr0_train1_model_dm_predictions.csv'

                    expected_value = len(pd.read_csv(resample_path + sub + '/tppredictions.csv')['x_pred'])
                    count += 1

                tp_x = np.array(pd.read_csv(resample_path + sub + '/gsr0_train1_model_tp_predictions.csv')['x_pred'])
                tp_y = np.array(pd.read_csv(resample_path + sub + '/gsr0_train1_model_tp_predictions.csv')['y_pred'])
                dm_x = np.array(
                    pd.read_csv(resample_path + sub + '/gsr0_train1_model_dm_predictions.csv')['x_pred'][:250])
                dm_y = np.array(
                    pd.read_csv(resample_path + sub + '/gsr0_train1_model_dm_predictions.csv')['y_pred'][:250])

                if (len(tp_x) == expected_value) & (len(dm_x) == expected_value):
                    corr_matrix_tp_x.append(tp_x)
                    corr_matrix_dm_x.append(tp_y)
                    corr_matrix_tp_y.append(dm_x)
                    corr_matrix_dm_y.append(dm_y)

            except:

                continue

        corr_matrix_x = np.corrcoef(corr_matrix_tp_x, corr_matrix_dm_x)
        corr_matrix_y = np.corrcoef(corr_matrix_tp_y, corr_matrix_dm_y)

        # Correction for Visualization of Correlation Matrix - OPTIONAL

        # for input_matrix in [corr_matrix_x, corr_matrix_y]:
        #
        #     anti_corr_list = []
        #     corr_list = []
        #
        #     for item1 in range(len(input_matrix)):
        #         for item2 in range(len(input_matrix[0])):
        #             if input_matrix[item1][item2] < 0:
        #                 anti_corr_list.append(input_matrix[item1][item2])
        #                 input_matrix[item1][item2] = 0
        #             else:
        #                 corr_list.append(input_matrix[item1][item2])
        #     print(len(anti_corr_list), len(corr_list))

        return corr_matrix_x, corr_matrix_y, corr_matrix_tp_x, corr_matrix_tp_y, corr_matrix_dm_x, corr_matrix_dm_y

    corr_matrix_x, corr_matrix_y, tp_x, tp_y, dm_x, dm_y = create_corr_matrix(sub_list)

    svm_dict = {}

    for item in range(len(tp_x[0])):

        svm_dict[item] = []

    for fix in range(len(tp_x[0])):

        temp_list = []

        for item in range(len(tp_x)):

            temp_list.append(tp_x[item][fix])

        svm_dict[fix] = temp_list

    for fix in range(len(dm_x[0])):

        temp_list = []

        for item in range(len(dm_x)):

            temp_list.append(dm_x[item][fix])

        svm_dict[fix] = svm_dict[fix] + temp_list

    tp_labels = [0 for x in range(len(tp_x))]
    dm_labels = [1 for x in range(len(dm_x))]
    label_list = tp_labels + dm_labels

    svm_dict['labels'] = label_list

    df_x = pd.DataFrame.from_dict(svm_dict)

    svm_dict = {}

    for item in range(len(tp_y[0])):

        svm_dict[item] = []

    for fix in range(len(tp_y[0])):

        temp_list = []

        for item in range(len(tp_y)):

            temp_list.append(tp_y[item][fix])

        svm_dict[fix] = temp_list

    for fix in range(len(dm_y[0])):

        temp_list = []

        for item in range(len(dm_y)):

            temp_list.append(dm_y[item][fix])

        svm_dict[fix] = svm_dict[fix] + temp_list

    tp_labels = [0 for x in range(len(tp_y))]
    dm_labels = [1 for x in range(len(dm_y))]
    label_list = tp_labels + dm_labels

    svm_dict['labels'] = label_list

    df_y = pd.DataFrame.from_dict(svm_dict)

    train_set_x, test_set_x = train_test_split(df_x, test_size=tt_split)

    train_data_x = train_set_x.drop(['labels'], axis=1)
    test_data_x = test_set_x.drop(['labels'], axis=1)
    train_targets_x = train_set_x[['labels']]
    test_targets_x = test_set_x[['labels']]

    train_set_y, test_set_y = train_test_split(df_y, test_size=tt_split)

    train_data_y = train_set_y.drop(['labels'], axis=1)
    test_data_y = test_set_y.drop(['labels'], axis=1)
    train_targets_y = train_set_y[['labels']]
    test_targets_y = test_set_y[['labels']]

    clfx = svm.SVC(C=100, tol=.0001, kernel='linear', verbose=1, probability=True)
    clfy = svm.SVC(C=100, tol=.0001, kernel='linear', verbose=1, probability=True)

    clfx.fit(train_data_x, train_targets_x)
    predictions_x = clfx.predict(test_data_x)
    clfy.fit(train_data_y, train_targets_y)
    predictions_y = clfy.predict(test_data_y)\

    print('Confusion Matrix in x')
    print(confusion_matrix(predictions_x, test_targets_x))
    print('Confusion Matrix in y')
    print(confusion_matrix(predictions_y, test_targets_y))

    probas_x = clfx.predict_proba(test_data_x)
    probas_y = clfy.predict_proba(test_data_y)

    fpr_x, tpr_x, thresholds = roc_curve(test_targets_x, probas_x[:, 1])
    roc_auc_x = auc(fpr_x, tpr_x)

    fpr_y, tpr_y, thresholds = roc_curve(test_targets_y, probas_y[:, 1])
    roc_auc_y = auc(fpr_y, tpr_y)

    plt.figure(figsize=(5, 10))
    plt.subplot(211)
    plt.plot(fpr_x, tpr_x, label='AUROC = ' + str(roc_auc_x)[:6])
    plt.plot([0, 1], [0, 1], color='k', linestyle='--')
    plt.title('ROC Curve for linear SVM')
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.legend()

    plt.subplot(212)
    plt.plot(fpr_y, tpr_y, label='AUROC = ' + str(roc_auc_y)[:6])
    plt.plot([0, 1], [0, 1], color='k', linestyle='--')
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.legend()
    plt.savefig('/home/json/Desktop/peer_figures_final/roc_curves.png', dpi=600)
    plt.show()


def fig_nine():

    viewtype = 'tp'
    cscheme = 'inferno'

    def load_data(min_scan=2):
        """Returns list of subjects with at least the specified number of calibration scans

        :param min_scan: Minimum number of scans required to be included in subject list
        :return: Dataframe containing subject IDs, site of MRI collection, number of calibration scans, and motion measures
        :return: List containing subjects with at least min_scan calibration scans
        """

        params = pd.read_csv('model_outputs.csv', index_col='subject', dtype=object)
        params = params.convert_objects(convert_numeric=True)

        if min_scan == 2:

            params = params[(params.scan_count == 2) | (params.scan_count == 3)]

        elif min_scan == 3:

            params = params[params.scan_count == 3]

        sub_list = params.index.values.tolist()

        return params, sub_list


    params, sub_list = load_data(min_scan=2)

    x_stack = []
    y_stack = []
    sorted_by = 'mean_fd'

    params = params.sort_values(by=[sorted_by])
    sub_list = params.index.values.tolist()

    def create_sub_list_with_et_and_peer(full_list):

        """Creates a list of subjects with both ET and PEER predictions

        :param full_list: List of subject IDs containing all subjects with at least 2/3 valid calibration scans
        :return: Subject list with both ET and PEER predictions
        """

        et_list = []

        for sub in full_list:

            if (os.path.exists('/data2/Projects/Jake/Human_Brain_Mapping/' + sub + '/et_device_pred.csv')) and \
                    (os.path.exists(
                        '/data2/Projects/Jake/Human_Brain_Mapping/' + sub + '/gsr0_train1_model_tp_predictions.csv')):
                et_list.append(sub)

        return et_list

    et_list = create_sub_list_with_et_and_peer(sub_list)

    with open('/data2/Projects/Lei/Peers/scripts/data_check/TP_bad_sub.txt') as f:
        bad_subs_list_tp = f.readlines()
    with open('/data2/Projects/Lei/Peers/scripts/data_check/DM_bad_sub.txt') as f:
        bad_subs_list_dm = f.readlines()

    bad_subs_list_tp = [x.strip('\n') for x in bad_subs_list_tp]
    bad_subs_list_dm = [x.strip('\n') for x in bad_subs_list_dm]

    et_list = [x for x in et_list if x not in bad_subs_list_tp]
    et_list = [x for x in et_list if x not in bad_subs_list_dm]


    filename_dict = {'calibration': {'name': '/gsr0_train1_model_calibration_predictions.csv', 'num_vol': 135},
                     'tp': {'name': '/gsr0_train1_model_tp_predictions.csv', 'num_vol': 250},
                     'dm': {'name': '/gsr0_train1_model_dm_predictions.csv', 'num_vol': 750}}

    fixations = pd.read_csv('stim_vals.csv')
    x_targets = np.tile(np.repeat(np.array(fixations['pos_x']), 5) * monitor_width / 2, 1)
    y_targets = np.tile(np.repeat(np.array(fixations['pos_y']), 5) * monitor_height / 2, 1)

    modality = 'peer'

    for sub in sub_list:

        try:

            if modality == 'peer':

                temp_df = pd.DataFrame.from_csv(resample_path + sub + filename_dict[viewtype]['name'])

            elif modality == 'et':

                temp_df = pd.DataFrame.from_csv(
                    '/data2/Projects/Jake/Human_Brain_Mapping/' + sub + '/et_device_pred.csv')

            x_series = list(temp_df['x_pred'])
            y_series = list(temp_df['y_pred'])

            if (len(x_series) == filename_dict[viewtype]['num_vol']) and (
                    len(y_series) == filename_dict[viewtype]['num_vol']):
                x_series = [x if abs(x) < monitor_width / 2 + .1 * monitor_width else 0 for x in x_series]
                y_series = [x if abs(x) < monitor_height / 2 + .1 * monitor_height else 0 for x in y_series]

                x_stack.append(x_series)
                y_stack.append(y_series)

        except:

            print('Error processing subject ' + sub)

    arr = np.zeros(len(x_series))
    arrx = np.array([-np.round(monitor_width / 2, 0) for x in arr])
    arry = np.array([-np.round(monitor_height / 2, 0) for x in arr])

    if viewtype == 'calibration':

        for num in range(int(np.round(len(et_list) * .02, 0))):
            x_stack.append(arrx)
            y_stack.append(arry)

        for num in range(int(np.round(len(et_list) * .02, 0))):
            x_stack.append(x_targets)
            y_stack.append(y_targets)

    else:

        avg_series_x = np.mean(x_stack, axis=0)
        avg_series_y = np.mean(y_stack, axis=0)

        for num in range(int(np.round(len(et_list) * .05, 0))):
            x_stack.append(arrx)
            y_stack.append(arry)

        # for num in range(int(np.round(len(et_list) * .02, 0))):
        #     x_stack.append(avg_series_x)
        #     y_stack.append(avg_series_y)

    modality = 'et'

    retain_et_list = []

    count_include = 0
    count_exclude = 0

    for sub in sub_list:

        try:

            if modality == 'peer':

                temp_df = pd.DataFrame.from_csv(resample_path + sub + filename_dict[viewtype]['name'])

            elif modality == 'et':

                temp_df = pd.DataFrame.from_csv('/data2/Projects/Jake/Human_Brain_Mapping/' + sub + '/et_device_pred.csv')

            x_series = list(temp_df['x_pred'])
            y_series = list(temp_df['y_pred'])

            if (len(x_series) == filename_dict[viewtype]['num_vol']) and (
                    len(y_series) == filename_dict[viewtype]['num_vol']):
                x_series = [x if abs(x) < monitor_width / 2 + .1 * monitor_width else 0 for x in x_series]
                y_series = [x if abs(x) < monitor_height / 2 + .1 * monitor_height else 0 for x in y_series]

                x_count = x_series.count(-840)
                y_count = y_series.count(525.0)

                if (x_count < 13) and (y_count < 13):

                    count_include += 1

                    print(sub, count_include)
                    retain_et_list.append(sub)

                    x_stack.append(x_series)
                    y_stack.append(y_series)

                else:

                    count_exclude += 1

                    print('Subject ' + sub + ' excluded from analysis', count_exclude)

        except:

            continue

    x_hm = np.stack(x_stack)
    y_hm = np.stack(y_stack)

    x_spacing = len(x_hm[0])

    fig = plt.figure(figsize=(12, 10))
    plt.subplot(211)
    plt.title('Heatmap of The Present for Eye-Tracking and PEER')
    ax = sns.heatmap(x_hm, cmap=cscheme, yticklabels=False)
    ax.set(xlabel='TR')
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))
    ax.xaxis.set_major_locator(ticker.MultipleLocator(base=np.round(x_spacing / 5, 0)))
    plt.yticks([200, 505], ['PEER', 'Eye-Tracking'])

    plt.subplot(212)
    ax = sns.heatmap(y_hm, cmap=cscheme, yticklabels=False)
    ax.set(xlabel='TR')
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))
    ax.xaxis.set_major_locator(ticker.MultipleLocator(base=np.round(x_spacing / 5, 0)))
    plt.yticks([200, 505], ['PEER', 'Eye-Tracking'])
    plt.savefig('/home/json/Desktop/peer_figures_final/heatmap_et_peer.png', dpi=600)
    plt.show()














def fig_ten():

    df = pd.DataFrame.from_csv('/home/json/Desktop/peer/fingerprinting_qap.csv')
    df = df[(df.PEER1 <= .2) & (df.TP <=.2)]
    sub_list = df.index.values.tolist()

    def create_sub_list_with_et_and_peer(full_list):

        """Creates a list of subjects with both ET and PEER predictions

        :param full_list: List of subject IDs containing all subjects with at least 2/3 valid calibration scans
        :return: Subject list with both ET and PEER predictions
        """

        et_list = []

        for sub in full_list:

            if (os.path.exists('/data2/Projects/Jake/Human_Brain_Mapping/' + sub + '/et_device_pred.csv')) and \
                    (os.path.exists(
                        '/data2/Projects/Jake/Human_Brain_Mapping/' + sub + '/gsr0_train1_model_tp_predictions.csv')):
                et_list.append(sub)

        return et_list

    et_list = create_sub_list_with_et_and_peer(sub_list)

    with open('/data2/Projects/Lei/Peers/scripts/data_check/TP_bad_sub.txt') as f:
        bad_subs_list_tp = f.readlines()
    with open('/data2/Projects/Lei/Peers/scripts/data_check/DM_bad_sub.txt') as f:
        bad_subs_list_dm = f.readlines()

    bad_subs_list_tp = [x.strip('\n') for x in bad_subs_list_tp]
    bad_subs_list_dm = [x.strip('\n') for x in bad_subs_list_dm]

    et_list = [x for x in et_list if x not in bad_subs_list_tp]
    et_list = [x for x in et_list if x not in bad_subs_list_dm]

    def individual_series(peer_list, et_list):

        et_series = {}
        peer_series = {}

        for sub in peer_list:

            try:

                sub_df = pd.DataFrame.from_csv('/data2/Projects/Jake/Human_Brain_Mapping/' + sub + '/no_gsr_train1_tp_pred.csv')
                sub_x = sub_df['x_pred']
                sub_y = sub_df['y_pred']

                peer_series[sub] = {'x': sub_x, 'y': sub_y}

            except:

                print('Error with subject ' + sub)

        for sub in et_list:

            try:

                sub_df = pd.DataFrame.from_csv('/data2/Projects/Jake/Human_Brain_Mapping/' + sub + '/et_device_pred.csv')
                sub_x = sub_df['x_pred']
                sub_y = sub_df['y_pred']

                et_series[sub] = {'x': sub_x, 'y': sub_y}

            except:

                print('Error with subject ' + sub)

        return et_series, peer_series

    def med_series(peer_list, et_list):
        # peer_list = subjects with valid peer scans
        # et_list = subjects with valid et data

        et_series = {'x': [], 'y': []}
        peer_series = {'x': [], 'y': []}

        for sub in peer_list:

            try:

                sub_df = pd.DataFrame.from_csv('/data2/Projects/Jake/Human_Brain_Mapping/' + sub + '/no_gsr_train1_tp_pred.csv')
                sub_x = sub_df['x_pred']
                sub_y = sub_df['y_pred']

                if len(sub_x) == 250:
                    peer_series['x'].append(np.array(sub_x))
                    peer_series['y'].append(np.array(sub_y))

                else:
                    print(sub)

            except:

                print('Error with subject ' + sub)

        for sub in et_list:

            try:

                sub_df = pd.DataFrame.from_csv('/data2/Projects/Jake/Human_Brain_Mapping/' + sub + '/et_device_pred.csv')
                sub_x = sub_df['x_pred']
                sub_y = sub_df['y_pred']
                et_series['x'].append(np.array(sub_x))
                et_series['y'].append(np.array(sub_y))
            except:

                print('Error with subject ' + sub)

        et_mean_series = {'x': np.nanmedian(et_series['x'], axis=0), 'y': np.nanmedian(et_series['y'], axis=0)}
        peer_mean_series = {'x': np.nanmedian(peer_series['x'], axis=0), 'y': np.nanmedian(peer_series['y'], axis=0)}

        return et_mean_series, peer_mean_series

    et_individual_series, peer_individual_series = individual_series(sub_list, et_list)
    et_mean_series, peer_mean_series = med_series(sub_list, et_list)
    
    pd_et = pd.DataFrame.from_dict(et_mean_series)
    pd_peer = pd.DataFrame.from_dict(peer_mean_series)
    pd_et.to_csv('/home/json/Desktop/peer/et_group_median.csv')
    pd_peer.to_csv('/home/json/Desktop/peer/peer_group_median.csv')

    z_scored_x_peer = preprocessing.scale(peer_mean_series['x'])
    z_scored_x_et = preprocessing.scale(et_mean_series['x'])
    z_scored_y_peer = preprocessing.scale(peer_mean_series['y'])
    z_scored_y_et = preprocessing.scale(et_mean_series['y'])

    no_z_x = spearmanr(peer_mean_series['x'], et_mean_series['x'])[0]
    no_z_y = spearmanr(peer_mean_series['y'], et_mean_series['y'])[0]

    x_corr_val = spearmanr(z_scored_x_peer, z_scored_x_et)[0]
    y_corr_val = spearmanr(z_scored_y_peer, z_scored_y_et)[0]
    
    x_axis = range(len(peer_mean_series['x']))

    plt.figure(figsize=(18, 12))
    plt.subplot(221)
    plt.title('Median fixation series for ET and PEER')
    plt.plot(x_axis, peer_mean_series['x'], 'r-', label=('PEER, r=' + str(no_z_x)[:5]), alpha=.75)
    plt.plot(x_axis, et_mean_series['x'], 'b-', label='ET', alpha=.75)
    plt.legend(loc=1)
    plt.subplot(223)
    plt.plot(x_axis, peer_mean_series['y'], 'r-', label=('PEER, r=' + str(no_z_y)[:5]), alpha=.75)
    plt.plot(x_axis, et_mean_series['y'], 'b-', label='ET', alpha=.75)
    plt.legend(loc=1)
    plt.subplot(222)
    plt.title('Z-centered fixation series for ET and PEER')
    plt.plot(x_axis, z_scored_x_peer, 'r-', label=('PEER, r= ' + str(x_corr_val)[:5]), alpha=.75)
    plt.plot(x_axis, z_scored_x_et, 'b-', label='ET', alpha=.75)
    plt.legend(loc=1)
    plt.subplot(224)
    plt.plot(x_axis, z_scored_y_peer, 'r-', label=('PEER, r= ' + str(y_corr_val)[:5]), alpha=.75)
    plt.plot(x_axis, z_scored_y_et, 'b-', label='ET', alpha=.75)
    plt.legend(loc=1)
    #plt.savefig('/home/json/Desktop/peer_figures_final/median_et_peer_comparison_spearman.png', dpi=600)
    plt.show()

########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################


def diversity_score_analysis(plot_status=True, mod='et'):

    def plot_heatmap_from_stacked_fixation_series(fixation_series, viewtype, direc='x'):

        """Plots heatmaps based on fixation series

        :param fixation_series: Numpy array containing stacked fixation series
        :param viewtype: Viewing stimulus
        :param direc: x- or y- direction specification for figure title
        :return: Heatmap of stacked fixation series
        """

        x_spacing = len(fixation_series[0])

        # sns.set()
        # plt.clf()
        # ax = sns.heatmap(fixation_series)
        # ax.set(xlabel='Volumes', ylabel='Subjects')
        # ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))
        # ax.xaxis.set_major_locator(ticker.MultipleLocator(base=np.round(x_spacing / 5, 0)))
        # ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))
        # ax.yaxis.set_major_locator(ticker.MultipleLocator(base=100))
        # plt.title('Fixation Series for ' + viewtype + ' in ' + direc)
        # plt.show()

    def stack_fixation_series(viewtype='calibration', sorted_by='mean_fd', modality='peer'):

        """ Stacks fixations for a given viewtype for heatmap visualization

        :param params: Dataframe that contains subject IDs and motion measures
        :param viewtype: Viewing stimulus
        :param sorted_by: Sort by mean_fd or dvars
        :return: Heatmap for x- and y- directions for a given viewtype
        """

        monitor_width = 1680
        monitor_height = 1050

        x_stack = []
        y_stack = []

        df = pd.DataFrame.from_csv('/home/json/Desktop/peer/et_qap.csv')
        df = df.set_index('Subjects')
        sub_list = df.index.values.tolist()
    
        filename_dict = {'calibration': {'name': '/gsr0_train1_model_calibration_predictions.csv', 'num_vol': 135},
                         'tp': {'name': '/gsr0_train1_model_tp_predictions.csv', 'num_vol': 250},
                         'dm': {'name': '/gsr0_train1_model_dm_predictions.csv', 'num_vol': 750}}

        fixations = pd.read_csv('/home/json/Desktop/peer/stim_vals.csv')
        x_targets = np.tile(np.repeat(np.array(fixations['pos_x']), 5) * monitor_width / 2, 1)
        y_targets = np.tile(np.repeat(np.array(fixations['pos_y']), 5) * monitor_height / 2, 1)


        for sub in sub_list:

            try:

                if modality == 'peer':

                    temp_df = pd.DataFrame.from_csv(resample_path + sub + filename_dict[viewtype]['name'])

                elif modality == 'et':

                    temp_df = pd.DataFrame.from_csv(
                        '/data2/Projects/Jake/Human_Brain_Mapping/' + sub + '/et_device_pred.csv')

                x_series = list(temp_df['x_pred'])
                y_series = list(temp_df['y_pred'])

                if (len(x_series) == filename_dict[viewtype]['num_vol']) and (
                        len(y_series) == filename_dict[viewtype]['num_vol']):
                    x_series = [x if abs(x) < monitor_width / 2 + .1 * monitor_width else 0 for x in x_series]
                    y_series = [x if abs(x) < monitor_height / 2 + .1 * monitor_height else 0 for x in y_series]

                    x_stack.append(x_series)
                    y_stack.append(y_series)

                else:
                
                    print('Subject ' + sub + ' excluded from analysis')

            except:

                continue

        x_hm_without_mean = np.stack(x_stack)
        y_hm_without_mean = np.stack(y_stack)

        arr = np.zeros(len(x_series))
        arrx = np.array([-np.round(monitor_width / 2, 0) for x in arr])
        arry = np.array([-np.round(monitor_height / 2, 0) for x in arr])

        if viewtype == 'calibration':

            for num in range(int(np.round(len(sub_list) * .02, 0))):
                x_stack.append(arrx)
                y_stack.append(arry)

            for num in range(int(np.round(len(sub_list) * .02, 0))):
                x_stack.append(x_targets)
                y_stack.append(y_targets)

        else:

            avg_series_x = np.mean(x_stack, axis=0)
            avg_series_y = np.mean(y_stack, axis=0)

            for num in range(int(np.round(len(sub_list) * .02, 0))):
                x_stack.append(arrx)
                y_stack.append(arry)

            for num in range(int(np.round(len(sub_list) * .02, 0))):
                x_stack.append(avg_series_x)
                y_stack.append(avg_series_y)

        x_hm = np.stack(x_stack)
        y_hm = np.stack(y_stack)

        plot_heatmap_from_stacked_fixation_series(x_hm, viewtype, direc='x')
        plot_heatmap_from_stacked_fixation_series(y_hm, viewtype, direc='y')

        return x_hm, y_hm, x_hm_without_mean, y_hm_without_mean

    monitor_width = 1680
    monitor_height = 1050

    uniform_val = float(1 / ((monitor_width / 30) * (monitor_height / 30)))

    x_hm, y_hm, x_hm_without_end, y_hm_without_end = stack_fixation_series(viewtype='tp', sorted_by='mean_fd', modality=mod)

    fixation_bins_vols = {}

    for vol in range(len(x_hm_without_end[0])):

        fixation_bins_vols[str(vol)] = []

    for vol in range(len(x_hm_without_end[0])):

        fixation_bins_vols[str(vol)] = {str(bin1): {str(bin2): [] for bin2 in range(int(monitor_height / 30))} \
                                        for bin1 in range(int(monitor_width / 30))}

    fixation_series = {}

    for vol in range(len(x_hm_without_end[0])):

        fixation_series[vol] = []

    for vol in range(len(x_hm_without_end[0])):

        for sub in range(len(x_hm_without_end)):

            fixation_series[vol].append([x_hm_without_end[sub][vol], y_hm_without_end[sub][vol]])

    out_of_bounds = {}
    sum_stats_per_vol = {}

    for vol in range(len(x_hm_without_end[0])):

        out_of_bounds_count = 0
        out_of_bounds[vol] = []

        total_count = 0
        present_in_vol = 0

        for sub in range(len(x_hm_without_end)):

            x_bin = str(int(np.floor((x_hm_without_end[sub][vol] + 840)/30)))
            y_bin = str(int(np.floor((y_hm_without_end[sub][vol] + 525)/30)))

            if (int(x_bin) < 0) or (int(x_bin) > 55) or (int(y_bin) < 0) or (int(y_bin) > 34):

                out_of_bounds_count += 1
                total_count += 1

            else:

                fixation_bins_vols[str(vol)][x_bin][y_bin].append('fixation')
                total_count += 1
                present_in_vol += 1

        out_of_bounds[vol] = out_of_bounds_count
        expected_count = present_in_vol + out_of_bounds_count

        sum_stats_per_vol[str(vol)] = {'total count': int(total_count), 'expected count': int(expected_count),
                                       'out of bounds count': int(out_of_bounds_count), 'fixation count': int(present_in_vol)}

    for vol in range(len(x_hm_without_end[0])):

        for bin1 in fixation_bins_vols[str(vol)].keys():

            for bin2 in fixation_bins_vols[str(vol)][bin1].keys():

                fixation_bins_vols[str(vol)][bin1][bin2] = abs((len(fixation_bins_vols[str(vol)][bin1][bin2]) / sum_stats_per_vol[str(vol)]['fixation count']) - uniform_val)


    for vol in fixation_bins_vols.keys():
        prop_validation = 0
        for bin1 in fixation_bins_vols[vol].keys():
            for bin2 in fixation_bins_vols[vol][bin1].keys():
                prop_validation += fixation_bins_vols[vol][bin1][bin2]

    diversity_score_dict = {}

    for vol in range(len(x_hm_without_end[0])):

        count_val = 0

        for bin1 in fixation_bins_vols[str(vol)].keys():

            for bin2 in fixation_bins_vols[str(vol)]['0'].keys():

                count_val += fixation_bins_vols[str(vol)][bin1][bin2]

        diversity_score_dict[str(vol)] = count_val

    div_scores_list = list(diversity_score_dict.values())

    rank_scores_list = [percentileofscore(div_scores_list, x) for x in div_scores_list]
    # rank_scores_list = preprocessing.scale(div_scores_list)

    if plot_status:

        ind_ax = np.linspace(0, len(div_scores_list)-1, len(div_scores_list))

        plt.figure(figsize=(10, 15))
        plt.tight_layout(h_pad=3)
        plt.subplot(311)
        plt.plot(ind_ax, div_scores_list)
        plt.title('Diversity Scores for The Present')
        plt.ylabel('Raw Score')

        plt.subplot(312)
        plt.plot(ind_ax, rank_scores_list)
        plt.ylabel('Percentile')

        plt.subplot(313)
        plt.plot(ind_ax, out_of_bounds.values())
        plt.ylabel('# Fixations Lost')
        plt.xlabel('Volumes (TR)')
        plt.show()

    output = zip(np.linspace(0, len(rank_scores_list)-1, len(rank_scores_list)), rank_scores_list)
    
    rank_dict = {'q1': [], 'q2': [], 'q3': [], 'tercile': []}

    for vol, rank in output:
        
        if rank <= 33:
            rank_dict['q1'].append(vol)
            rank_dict['tercile'].append('q1')

        elif 33 < rank <= 67:
            rank_dict['q2'].append(vol)
            rank_dict['tercile'].append('q2')
            
        elif rank > 67:
            rank_dict['q3'].append(vol)
            rank_dict['tercile'].append('q3')

    return rank_dict, sum_stats_per_vol, div_scores_list, rank_scores_list, out_of_bounds.values()

sns.set()

rank_dict1, sum_stats_per_vol1, div_scores_list1, rank_scores_list1, out_of_bounds_list1 = diversity_score_analysis(
    plot_status=False, mod='peer')

rank_dict2, sum_stats_per_vol2, div_scores_list2, rank_scores_list2, out_of_bounds_list2 = diversity_score_analysis(
    plot_status=False, mod='et')
    
output_dict = {'tercile': rank_dict2['tercile'], 'volume': rank_dict2['q1'] + rank_dict2['q2'] + rank_dict2['q3']}
df_out = pd.DataFrame.from_dict(output_dict)
df_out.to_csv('/home/json/Desktop/peer/et_tercile_diversity_score.csv')

ind_ax = np.linspace(0, len(div_scores_list1) - 1, len(div_scores_list1))

plt.figure()
plt.plot(ind_ax, rank_scores_list1, label='PEER')
plt.plot(ind_ax, rank_scores_list2, label='ET')
plt.legend()
plt.show()

print(pearsonr(rank_scores_list1, rank_scores_list2)[0])



joint_div_scores = zip(rank_scores_list1, rank_scores_list2)

vol_index = 0

for item1, item2 in joint_div_scores:

    if (item1 < 5) and (item2) < 5:

        print('Volume ' + str(vol_index) + ' has a diversity score < 5')

    elif (item1 > 90) and (item2) >  90:

        print('Volume ' + str(vol_index) + ' has a diversity score > 90')

    vol_index += 1






########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################


def fig_twelve():
    
    df = pd.DataFrame.from_csv('/home/json/Desktop/peer/et_qap.csv')
    df = df.set_index('Subjects')
    et_list = df.index.values.tolist()
    
    within_corr_x = []
    within_corr_y = []
    
    combinations_list = []
    
    test_dict = {}
    
    for sub in et_list:
        
        try:
            
            et_df = pd.DataFrame.from_csv('/data2/Projects/Jake/Human_Brain_Mapping/' + sub + '/et_device_pred.csv')
            et_x = et_df['x_pred']
            et_y = et_df['y_pred']
            
            peer_df = pd.DataFrame.from_csv('/data2/Projects/Jake/Human_Brain_Mapping/' + sub + '/gsr0_train1_model_tp_predictions.csv')
            peer_x = peer_df['x_pred']
            peer_y = peer_df['y_pred']
            
            x_corr = pearsonr(et_x, peer_x)[0]
            y_corr = pearsonr(et_y, peer_y)[0]
            
            if (np.isnan(x_corr)) or (np.isnan(y_corr)):
                
                print('Bad ET data for subject ' + sub)
                
            else:
            
                within_corr_x.append(x_corr)
                within_corr_y.append(y_corr)
                combinations_list.append([sub])
                test_dict[sub] = x_corr
            
        except:
            
            continue

    unique_subject_combinations = combinations(combinations_list, 2)
    
    bn_corr_x = []
    bn_corr_y = []
    
    count = 0
    count_max = math.factorial(len(combinations_list)) / (math.factorial(len(combinations_list) - 2) * 2)
    
    for item1, item2 in unique_subject_combinations:
        
        et_df = pd.DataFrame.from_csv('/data2/Projects/Jake/Human_Brain_Mapping/' + item1[0] + '/et_device_pred.csv')
        et_x = et_df['x_pred']
        et_y = et_df['y_pred']
            
        peer_df = pd.DataFrame.from_csv('/data2/Projects/Jake/Human_Brain_Mapping/' + item2[0] + '/gsr0_train1_model_tp_predictions.csv')
        peer_x = peer_df['x_pred']
        peer_y = peer_df['y_pred']
        
        x_corr = spearmanr(et_x, peer_x)[0]
        y_corr = spearmanr(et_y, peer_y)[0]        
        
        bn_corr_x.append(x_corr)
        bn_corr_y.append(y_corr)
        
        et_df = pd.DataFrame.from_csv('/data2/Projects/Jake/Human_Brain_Mapping/' + item2[0] + '/et_device_pred.csv')
        et_x = et_df['x_pred']
        et_y = et_df['y_pred']
            
        peer_df = pd.DataFrame.from_csv('/data2/Projects/Jake/Human_Brain_Mapping/' + item1[0] + '/gsr0_train1_model_tp_predictions.csv')
        peer_x = peer_df['x_pred']
        peer_y = peer_df['y_pred']
        
        x_corr = spearmanr(et_x, peer_x)[0]
        y_corr = spearmanr(et_y, peer_y)[0]        
        
        bn_corr_x.append(x_corr)
        bn_corr_y.append(y_corr)
        
        count += 1
        
        if count in [np.floor(x* count_max) for x in np.linspace(0, 1, 21)]:
            print(str(count*100/count_max)[:4] + '% complete')
    
    violin_dict = {'Comparison': ['Within']*len(within_corr_x)*2 + ['Between']*len(bn_corr_x)*2,
    'Correlations': within_corr_x + within_corr_y + bn_corr_x + bn_corr_y,
    'Direction': ['x']*len(within_corr_x) + ['y']*len(within_corr_x) + ['x']*len(bn_corr_x) + ['y']*len(bn_corr_x)}
    
    violin_df = pd.DataFrame.from_dict(violin_dict)
    
    sns.set()
    sns.violinplot(x='Direction', y='Correlations', hue='Comparison', data=violin_df)
#    plt.savefig('/home/json/Desktop/peer_figures_final/fingerprinting.png', dpi=600)
    plt.show()

n, bins = np.histogram(within_corr_x, 20, density=1)
pdfx = np.zeros(n.size)
pdfy = np.zeros(n.size)
for k in range(n.size):
    pdfx[k] = 0.5*(bins[k]+bins[k+1])
    pdfy[k] = n[k]

plt.plot(pdfx, pdfy / np.sum(pdfy), label='Within x')

n, bins = np.histogram(bn_corr_x, 20, density=1)
pdfx = np.zeros(n.size)
pdfy = np.zeros(n.size)
for k in range(n.size):
    pdfx[k] = 0.5*(bins[k]+bins[k+1])
    pdfy[k] = n[k]

plt.plot(pdfx, pdfy / np.sum(pdfy), label='Between x')
plt.xlabel("Pearson's r")
plt.ylabel("Proportion")
plt.legend()
#plt.savefig('/home/json/Desktop/peer_figures_final/fingerprint_comparison_dist.png', dpi=600)
plt.show()


def fingerprinting_qap():
    
    qap_dict = {'Subject': [], 'PEER1': [], 'TP': []}
    
    params, sub_list = load_data(min_scan=2)
    
    qap_df = pd.DataFrame.from_csv('/data2/HBNcore/CMI_HBN_Data/MRI/RU/QAP/qap_functional_temporal.csv')
    
    for sub in sub_list:
        
        try:
            
            sub = sub.replace('-', '_')    
            
            peer1_meanfd = list(qap_df[(qap_df.Participant == sub) & (qap_df.Series == 'func_peer_run_1')]['RMSD (Mean)'])[0]
            tp_meanfd = list(qap_df[(qap_df.Participant == sub) & (qap_df.Series == 'func_movieTP')]['RMSD (Mean)'])[0]
            
            qap_dict['Subject'].append(sub.replace('_', '-'))
            qap_dict['PEER1'].append(peer1_meanfd)
            qap_dict['TP'].append(tp_meanfd)
            
        except:
            
            print('Subject ' + sub + ' not completed')
            
    df = pd.DataFrame.from_dict(qap_dict)
    df = df.set_index('Subject')
    
    df.to_csv('/home/json/Desktop/peer/fingerprinting_qap.csv')
    
def get_list_of_subjects_with_good_data_mri():

    df = pd.DataFrame.from_csv('/home/json/Desktop/peer/fingerprinting_qap.csv')
    df = df[(df.PEER1 <= .2) & (df.TP <=.2)]
    sub_list = df.index.values.tolist()
    
    return sub_list
    




def testing_tercile_within_and_bn_comparison(mod='peer', tercile='q1'):
    
    df = pd.DataFrame.from_csv('/home/json/Desktop/peer/et_qap.csv')
    df = df.set_index('Subjects')
    et_list = df.index.values.tolist()
    
    if mod == 'peer':
        df_terciles = pd.DataFrame.from_csv('/home/json/Desktop/peer/peer_tercile_diversity_score.csv')
    elif mod == 'et':
        df_terciles = pd.DataFrame.from_csv('/home/json/Desktop/peer/et_tercile_diversity_score.csv')
    df_terciles = df_terciles[df_terciles.tercile == tercile]
    volume_selection = list(df_terciles.volume)
    volume_selection = [int(x) for x in volume_selection]
    
    within_corr_x = []
    within_corr_y = []
    
    combinations_list = []
    
    test_dict = {}
    
    for sub in et_list:
        
        try:
            
            et_df = pd.DataFrame.from_csv('/data2/Projects/Jake/Human_Brain_Mapping/' + sub + '/et_device_pred.csv')
            et_x = et_df['x_pred']
            et_y = et_df['y_pred']
            
            peer_df = pd.DataFrame.from_csv('/data2/Projects/Jake/Human_Brain_Mapping/' + sub + '/gsr0_train1_model_tp_predictions.csv')
            peer_x = peer_df['x_pred']
            peer_y = peer_df['y_pred']
            
            et_x = [et_x[x] for x in volume_selection]
            et_y = [et_y[x] for x in volume_selection]
            peer_x = [peer_x[x] for x in volume_selection]
            peer_y = [peer_y[x] for x in volume_selection]
            
            x_corr = pearsonr(et_x, peer_x)[0]
            y_corr = pearsonr(et_y, peer_y)[0]
            
            if (np.isnan(x_corr)) or (np.isnan(y_corr)):
                
                print('Bad ET data for subject ' + sub)
                
            else:
            
                within_corr_x.append(x_corr)
                within_corr_y.append(y_corr)
                combinations_list.append([sub])
                test_dict[sub] = x_corr
            
        except:
            
            continue

    unique_subject_combinations = combinations(combinations_list, 2)
    
    bn_corr_x = []
    bn_corr_y = []
    
    count = 0
    count_max = math.factorial(len(combinations_list)) / (math.factorial(len(combinations_list) - 2) * 2)
    
    for item1, item2 in unique_subject_combinations:
        
        et_df = pd.DataFrame.from_csv('/data2/Projects/Jake/Human_Brain_Mapping/' + item1[0] + '/et_device_pred.csv')
        et_x = et_df['x_pred']
        et_y = et_df['y_pred']
            
        peer_df = pd.DataFrame.from_csv('/data2/Projects/Jake/Human_Brain_Mapping/' + item2[0] + '/gsr0_train1_model_tp_predictions.csv')
        peer_x = peer_df['x_pred']
        peer_y = peer_df['y_pred']
        
        et_x = [et_x[x] for x in volume_selection]
        et_y = [et_y[x] for x in volume_selection]
        peer_x = [peer_x[x] for x in volume_selection]
        peer_y = [peer_y[x] for x in volume_selection]
        
        x_corr = spearmanr(et_x, peer_x)[0]
        y_corr = spearmanr(et_y, peer_y)[0]        
        
        bn_corr_x.append(x_corr)
        bn_corr_y.append(y_corr)
        
        et_df = pd.DataFrame.from_csv('/data2/Projects/Jake/Human_Brain_Mapping/' + item2[0] + '/et_device_pred.csv')
        et_x = et_df['x_pred']
        et_y = et_df['y_pred']
            
        peer_df = pd.DataFrame.from_csv('/data2/Projects/Jake/Human_Brain_Mapping/' + item1[0] + '/gsr0_train1_model_tp_predictions.csv')
        peer_x = peer_df['x_pred']
        peer_y = peer_df['y_pred']
        
        et_x = [et_x[x] for x in volume_selection]
        et_y = [et_y[x] for x in volume_selection]
        peer_x = [peer_x[x] for x in volume_selection]
        peer_y = [peer_y[x] for x in volume_selection]
        
        x_corr = spearmanr(et_x, peer_x)[0]
        y_corr = spearmanr(et_y, peer_y)[0]        
        
        bn_corr_x.append(x_corr)
        bn_corr_y.append(y_corr)
        
        count += 1
        
        if count in [np.floor(x* count_max) for x in np.linspace(0, 1, 21)]:
            print(str(count*100/count_max)[:4] + '% complete')
    
    violin_dict = {'Comparison': ['Within']*len(within_corr_x)*2 + ['Between']*len(bn_corr_x)*2,
    'Correlations': within_corr_x + within_corr_y + bn_corr_x + bn_corr_y,
    'Direction': ['x']*len(within_corr_x) + ['y']*len(within_corr_x) + ['x']*len(bn_corr_x) + ['y']*len(bn_corr_x)}
    
    violin_df = pd.DataFrame.from_dict(violin_dict)
    
    sns.set()
    sns.violinplot(x='Direction', y='Correlations', hue='Comparison', data=violin_df)
    plt.ylim([-1, 1])
    plt.savefig('/home/json/Desktop/peer_figures_final/' + mod + tercile.strip('q') + '.png', dpi=600)
    plt.clf()





def difference_series_svm():
    
    df = pd.DataFrame.from_csv('/home/json/Desktop/peer/et_qap.csv')
    df = df.set_index('Subjects')
    et_list = df.index.values.tolist()
    
    qap_list = []    
    
    for sub in et_list:
        
        try:
        
            et = pd.DataFrame.from_csv('/data2/Projects/Jake/Human_Brain_Mapping/' + sub + '/et_device_pred.csv')['x_pred']
            pe = pd.DataFrame.from_csv('/data2/Projects/Jake/Human_Brain_Mapping/' + sub + '/gsr0_train1_model_tp_predictions.csv')['y_pred']
        
            if len(et) == len(pe) == 250:
                
                qap_list.append(sub)
            
            else:
            
                print('Subject ' + sub + ' excluded')
                
        except:
            
            print('Subject ' + sub + ' excluded')
    
    et_list = np.array(qap_list)
    
    kf = KFold(n_splits=5)
    
    count = 1
    
    for train_index, test_index in kf.split(et_list):
        
        print('Validation set ' + str(count))
        
        train_subjects = [str(x) for x in et_list[train_index]]
        test_subjects = [str(x) for x in et_list[test_index]]
        
        train_list = []
        test_list = []
        
        for sub in train_subjects:
            
            et_df = pd.DataFrame.from_csv('/data2/Projects/Jake/Human_Brain_Mapping/' + sub + '/et_device_pred.csv')
            pe_df = pd.DataFrame.from_csv('/data2/Projects/Jake/Human_Brain_Mapping/' + sub + '/gsr0_train1_model_tp_predictions.csv')
            et_x = preprocessing.scale(et_df['x_pred'])
            et_y = preprocessing.scale(et_df['y_pred'])
            pe_x = preprocessing.scale(pe_df['x_pred'])
            pe_y = preprocessing.scale(pe_df['y_pred'])
            
            diff_x = et_x - pe_x
            diff_y = et_y - pe_y
            
            same_sub = list(diff_x) + list(diff_y) + ['within']
            train_list.append(same_sub)
        
        for sub in test_subjects:
            
            et_df = pd.DataFrame.from_csv('/data2/Projects/Jake/Human_Brain_Mapping/' + sub + '/et_device_pred.csv')
            pe_df = pd.DataFrame.from_csv('/data2/Projects/Jake/Human_Brain_Mapping/' + sub + '/gsr0_train1_model_tp_predictions.csv')
            et_x = preprocessing.scale(et_df['x_pred'])
            et_y = preprocessing.scale(et_df['y_pred'])
            pe_x = preprocessing.scale(pe_df['x_pred'])
            pe_y = preprocessing.scale(pe_df['y_pred'])
            
            diff_x = et_x - pe_x
            diff_y = et_y - pe_y
            
            same_sub = list(diff_x) + list(diff_y) + ['within']
            test_list.append(same_sub)    
        
        
        train_combinations = combinations([[x] for x in train_subjects], 2)
        test_combinations = combinations([[x] for x in test_subjects], 2)
        
        for item1, item2 in train_combinations:
            
            et_df = pd.DataFrame.from_csv('/data2/Projects/Jake/Human_Brain_Mapping/' + item1[0] + '/et_device_pred.csv')
            peer_df = pd.DataFrame.from_csv('/data2/Projects/Jake/Human_Brain_Mapping/' + item2[0] + '/gsr0_train1_model_tp_predictions.csv')
            
            et_x = et_df['x_pred']
            et_y = et_df['y_pred']
            peer_x = peer_df['x_pred']
            peer_y = peer_df['y_pred']
            et_x = preprocessing.scale(et_x)
            et_y = preprocessing.scale(et_y)
            pe_x = preprocessing.scale(peer_x)
            pe_y = preprocessing.scale(peer_y)
            
            diff_x = et_x - pe_x
            diff_y = et_y - pe_y
            
            diff_sub = list(diff_x) + list(diff_y) + ['between']
            train_list.append(diff_sub)
        
        for item1, item2 in test_combinations:
            
            et_df = pd.DataFrame.from_csv('/data2/Projects/Jake/Human_Brain_Mapping/' + item1[0] + '/et_device_pred.csv')
            peer_df = pd.DataFrame.from_csv('/data2/Projects/Jake/Human_Brain_Mapping/' + item2[0] + '/gsr0_train1_model_tp_predictions.csv')
            
            et_x = et_df['x_pred']
            et_y = et_df['y_pred']
            peer_x = peer_df['x_pred']
            peer_y = peer_df['y_pred']
            et_x = preprocessing.scale(et_x)
            et_y = preprocessing.scale(et_y)
            pe_x = preprocessing.scale(peer_x)
            pe_y = preprocessing.scale(peer_y)
            
            diff_x = et_x - pe_x
            diff_y = et_y - pe_y
            
            diff_sub = list(diff_x) + list(diff_y) + ['between']
            test_list.append(diff_sub)
            
        train_df = pd.DataFrame(train_list[:300])
        test_df = pd.DataFrame(test_list[:100])
        train_labels = np.array(train_df[500].tolist())
        test_labels = np.array(test_df[500].tolist())
        train_df = train_df.drop([500], axis=1)
        test_df = test_df.drop([500], axis=1)
        
        # MESS AROUND WITH SVM HYPERPARAMETERS
        
        clf = svm.SVC(kernel='rbf', tol=1e-3, class_weight={'between': 10})
        clf.fit(train_df, train_labels)
        predictions = clf.predict(test_df)
        print(accuracy_score(predictions, test_labels))
        print(confusion_matrix(predictions, test_labels))
        
        count += 1

        
    


























