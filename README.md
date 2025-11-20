# Docker Workspace Template

## How to install

There are two ways to setup this repo:

**A)** As **single folder** project (only this one git repo)
- *Recommended for single python projects*  
- Create a new folder for your package in the `./project/` directory
    ```bash
    mkdir ./project/<my_package>
    ```
**B)** As **multi-repo** project which has a separate repo for the environment (this) + multiple other git repos.  
- *Recommended for ROS development*
- Add all necessary repositories to the [`project.repos`](project.repos) file following the provided example format:
    ```yaml
    repositories:
        <folder name in ./project/>:
            type: git
            url: <git@repo_url>.git
            version: <branch>>
        <another folder name in ./project/>:
            #...
    ```
For either options, proceed with the following steps:

1. Run the installation script and select your desired project type:
    ```bash
    ./install.sh
    ```

2. (Optional) Cleanup your workspace and remove setup files:  
    *This will remove the entire `setup/` folder as well as the installation script.  
    If you plan to change the project type later, you might want to keep these files.*
    ```bash
    ./clean_setup_files.sh
    ```
3. Click on the "Build & Run" Task in the bottom row
4. Connect a new VS Code window to the running container and open the `/workspace/` folder.  
   This will load all recommended settings and extensions from the .vscode folder.  
   Accept the pop-up window to install recommended extensions.
5. Happy coding!


### Dependencies

The following tools are required to use this Docker workspace template:

- [VS Code](https://code.visualstudio.com/) - Primary development environment
- [Git](https://git-scm.com/) - Version control system
- [Docker](https://docs.docker.com/get-docker/) with [NVIDIA Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) - Container platform with GPU support
- [Python 3](https://www.python.org/downloads/) - Required for setup scripts
- [Python venv](https://docs.python.org/3/library/venv.html) - Virtual environment support

## How to use

This Docker workspace template provides a streamlined development environment with two main components:

- **Base Dockerfile**: Contains the pre-built foundation with common dependencies and tools
- **User Dockerfile**: Your customizable layer where you add project-specific requirements

> [!CAUTION]
> This tool runs all containers with `--privileged` flag for maximum compatibility!
> If you don't need privileged access, modify the `docker/run.sh` script accordingly for better security.

### Project Management

#### Using VS Code Tasks

The most convenient way to manage your project is through the **VS Code task buttons** located at the bottom of your editor. These provide one-click access to common operations like:

- Building Docker images
- Running containers
- Connecting to running containers
- Cleaning up Docker resources

#### Manual Configuration

For advanced users or automated workflows, you can **configure settings manually** by editing the `project.env` file. This file contains environment variables that control:

- Docker image names and tags
- Repository URLs and branches
- Build and runtime parameters
- Custom environment variables for your project

#### Project Structure

Your **project repositories will be automatically cloned** into the `project/` folder based on the repositories listed in `project.repos`. This project folder is **mounted as a volume** inside the Docker container, ensuring:

- Real-time synchronization between your host and container
- Persistent data that survives container restarts
- Seamless development workflow with your favorite IDE

## How to personalize

## How to contribute

**A)** To contribute a fix **from your specific `template-instance`** to this `template` repo, follow these steps:
1. Clone this `template` repo where you want to apply the change 
   ```bash
   git clone git@github.com:IRAS-HKA/docker_workspace_template.git
   cd docker_workspace_template
   ```
2. Add your `template-instance` as a new remote:
   ```bash
   git remote add template-instance git@<repo_url>.git
   git fetch template-instance
   ```
3. Cherry pick specific commits to apply them to this `template`:
   **Be very careful with commit selection to avoid conflicts!**
   ```bash
   git cherry-pick <commit-hash-from-your-template-instance-repo>
   ```
4. When done, you can remove the `template-instance` remote:
   ```bash
   git remote remove template-instance
   ```

**B)** To apply a fix **from this `template`** repo to your specific `template-instance`, follow these steps:
1. Switch in the repo where you want to apply the fix (`template-instance`)
   ```bash
   cd /path/to/your/template-instance
   ```
2. Add the `template` as a new remote:
   ```bash
   git remote add template git@github.com:IRAS-HKA/docker_workspace_template.git
   git fetch template
   ```
3. Cherry pick specific commits to apply them to your `template-instance`:
   **Be very careful with commit selection to avoid conflicts!**
   ```bash
   git cherry-pick <commit-hash-from-template-repo>
   ```
4. When done, you can remove the `template` remote:
   ```bash
   git remote remove template
   ```

## Open ToDo's

- [ ] TEST!
- [ ] Add support for git svn (currently only branches are allowed as versions)
- [ ] Make Dockerhub options more modular
- [ ] Implement dynamic README generation based on project type
    - [X] Basic stitching
    - [ ] Create Versions for each Projecttype
- [X] Create base image for PyTorch from source compilation
- [ ] Add standalone Python installation option
- [ ] Add standalone CUDA installation option
