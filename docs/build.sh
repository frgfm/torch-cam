function deploy_doc(){
    if [ ! -z "$1" ]
    then
        git checkout $1
    fi
    COMMIT=$(git rev-parse --short HEAD)
    echo "Creating doc at commit" $COMMIT "and pushing to folder $2"
    pip install -U ..
    if [ ! -z "$2" ]
    then
        if [ "$2" == "latest" ]; then
            echo "Pushing main"
            sphinx-build source build/$2 -a
        elif [ -d build/$2 ]; then
            echo "Directory" $2 "already exists"
        else
            echo "Pushing version" $2
            cp -r _static source/ && cp _conf.py source/conf.py
            sphinx-build source _build -a
        fi
    else
        echo "Pushing stable"
        cp -r _static source/ && cp _conf.py source/conf.py
        sphinx-build source build -a
    fi
    git checkout source/ && git clean -f source/
}

# exit when any command fails
set -e
# You can find the commit for each tag on https://github.com/frgfm/torch-cam/tags
if [ -d build ]; then rm -Rf build; fi
mkdir build
cp -r source/_static .
cp source/conf.py _conf.py
git fetch --all --tags --unshallow
deploy_doc "" latest
deploy_doc "eb9427e" v0.2.0
deploy_doc "d8d722d" v0.3.0
deploy_doc "e34fc42" v0.3.1
deploy_doc "1b6f37d" v0.3.2
deploy_doc "53e9dfe" # v0.4.0 Latest stable release
rm -rf _build _static _conf.py
