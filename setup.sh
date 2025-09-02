#!/usr/bin/env bash
action() {

    # Set main directories
    local shell_is_zsh="$( [ -z "${ZSH_VERSION}" ] && echo "false" || echo "true" )"
    local this_file="$( ${shell_is_zsh} && echo "${(%):-%x}" || echo "${BASH_SOURCE[0]}" )"
    local this_dir="$( cd "$( dirname "${this_file}" )" && pwd )"

    # set PYTHONPATH
    export PYTHONPATH="${this_dir}:${PYTHONPATH}"

    ENV_PATH=/global/cfs/cdirs/m3246/highD_PAWS/eventgen

    CONFIG_FILE="${this_dir}/.config"

    # Function to read the output directory from the config file
    read_config() {
        if [[ -f $CONFIG_FILE ]]; then
            source $CONFIG_FILE
        else
            export GEN_OUT=""
        fi
    }

    # Function to write the output directory to the config file
    write_config() {
        echo "export GEN_OUT=\"$GEN_OUT\"" > $CONFIG_FILE
    }

    # Prompt user for input if GEN_OUT is not set
    prompt_user() {
        read -p "Enter the output directory: " user_input
        if [[ -n $user_input ]]; then
            export GEN_OUT=$user_input
            write_config
        fi
    }

    # Main script execution
    read_config

    if [[ -z $GEN_OUT ]]; then
        echo "No output directory configured."
        prompt_user
    fi

    # Use the GEN_OUT in your script
    echo "Using output directory: $GEN_OUT"

    # Set code and law area
    export GEN_CODE="${this_dir}"
    export GEN_SLURM="${GEN_OUT}/slurm"

    export LAW_HOME="${this_dir}/.law"
    export LAW_CONFIG_FILE="${this_dir}/law.cfg"

    # Setup software directories
    export SOFTWARE_DIR="${this_dir}/software"
    mkdir -p $SOFTWARE_DIR

    # If no conda available, activate it
    if ! command -v conda >/dev/null 2>&1; then
        module load python
    fi

    # If conda env "eventgen" does not exist create it
    if ! conda info --envs | grep -q "$ENV_PATH"; then
        mamba create -p "$ENV_PATH"
        mamba env update -p "$ENV_PATH" --file eventgen.yml
    fi

    # Activate conda environment eventgen
    conda activate "$ENV_PATH"

    # law setup
    source "$( law completion )" ""


    export PYTHIA_DIR="${CONDA_PREFIX}"
    export DELPHES_DIR="${CONDA_PREFIX}"
}
action
