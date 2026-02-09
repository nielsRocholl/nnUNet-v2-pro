"""
Retired implementation of DefaultPreprocessor.run() method with original tqdm progress bar and print statements.
Preserved for reference.
"""


def run_original(self, dataset_name_or_id: Union[int, str], configuration_name: str, plans_identifier: str,
            num_processes: int):
    """
    data identifier = configuration name in plans. EZ.
    """
    dataset_name = maybe_convert_to_dataset_name(dataset_name_or_id)

    assert isdir(join(nnUNet_raw, dataset_name)), "The requested dataset could not be found in nnUNet_raw"

    plans_file = join(nnUNet_preprocessed, dataset_name, plans_identifier + '.json')
    assert isfile(plans_file), "Expected plans file (%s) not found. Run corresponding nnUNet_plan_experiment " \
                               "first." % plans_file
    plans = load_json(plans_file)
    plans_manager = PlansManager(plans)
    configuration_manager = plans_manager.get_configuration(configuration_name)

    if self.verbose:
        print(f'Preprocessing the following configuration: {configuration_name}')
    if self.verbose:
        print(configuration_manager)

    dataset_json_file = join(nnUNet_preprocessed, dataset_name, 'dataset.json')
    dataset_json = load_json(dataset_json_file)

    output_directory = join(nnUNet_preprocessed, dataset_name, configuration_manager.data_identifier)

    if isdir(output_directory):
        shutil.rmtree(output_directory)

    maybe_mkdir_p(output_directory)

    dataset = get_filenames_of_train_images_and_targets(join(nnUNet_raw, dataset_name), dataset_json)

    # identifiers = [os.path.basename(i[:-len(dataset_json['file_ending'])]) for i in seg_fnames]
    # output_filenames_truncated = [join(output_directory, i) for i in identifiers]

    # multiprocessing magic.
    r = []
    with multiprocessing.get_context("spawn").Pool(num_processes) as p:
        remaining = list(range(len(dataset)))
        # p is pretty nifti. If we kill workers they just respawn but don't do any work.
        # So we need to store the original pool of workers.
        workers = [j for j in p._pool]
        for k in dataset.keys():
            r.append(p.starmap_async(self.run_case_save,
                                     ((join(output_directory, k), dataset[k]['images'], dataset[k]['label'],
                                       plans_manager, configuration_manager,
                                       dataset_json),)))

        with tqdm(desc=None, total=len(dataset), disable=self.verbose) as pbar:
            while len(remaining) > 0:
                all_alive = all([j.is_alive() for j in workers])
                if not all_alive:
                    raise RuntimeError('Some background worker is 6 feet under. Yuck. \n'
                                       'OK jokes aside.\n'
                                       'One of your background processes is missing. This could be because of '
                                       'an error (look for an error message) or because it was killed '
                                       'by your OS due to running out of RAM. If you don\'t see '
                                       'an error message, out of RAM is likely the problem. In that case '
                                       'reducing the number of workers might help')
                done = [i for i in remaining if r[i].ready()]
                # get done so that errors can be raised
                _ = [r[i].get() for i in done]
                for _ in done:
                    r[_].get()  # allows triggering errors
                    pbar.update()
                remaining = [i for i in remaining if i not in done]
                sleep(0.1)
