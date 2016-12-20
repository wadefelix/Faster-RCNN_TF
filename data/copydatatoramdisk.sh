#!/bin/bash
# Usage:
# copydatatoramdisk.sh srcdir destname
#    such as: copydatatoramdisk.sh  /home/merge/data/JPGimageswithknife_RS/ VOC383
#       will makedir /dev/shm/VOC383 first, copy /home/merge/data/JPGimageswithknife_RS/* into it.
#       and, ln -s /data/VOCdevkit383 /dev/shm/VOC383

SRCDIR=$1
DESTDIRNAME=$2
DESTDIR=/dev/shm/$2
LNDESTDIR=~/py-faster-rcnn/data/${DESTDIRNAME/VOC/VOCdevkit}
PYFASTRCNNCACHE=~/py-faster-rcnn/data/cache

#####################################################################
# Print warning message.
function warning()
{
    echo "$*" >&2
}

#####################################################################
# Print error message and exit.
function error()
{
    echo "$*" >&2
    exit 1
}


#####################################################################
# Ask yesno question.
#
# Usage: yesno OPTIONS QUESTION
#
#   Options:
#     --timeout N    Timeout if no input seen in N seconds.
#     --default ANS  Use ANS as the default answer on timeout or
#                    if an empty answer is provided.
#
# Exit status is the answer.
function yesno()
{
    local ans
    local ok=0
    local timeout=0
    local default
    local t

    while [[ "$1" ]]
    do
        case "$1" in
        --default)
            shift
            default=$1
            if [[ ! "$default" ]]; then error "Missing default value"; fi
            t=$(tr '[:upper:]' '[:lower:]' <<<$default)

            if [[ "$t" != 'y'  &&  "$t" != 'yes'  &&  "$t" != 'n'  &&  "$t" != 'no' ]]; then
                error "Illegal default answer: $default"
            fi
            default=$t
            shift
            ;;

        --timeout)
            shift
            timeout=$1
            if [[ ! "$timeout" ]]; then error "Missing timeout value"; fi
            if [[ ! "$timeout" =~ ^[0-9][0-9]*$ ]]; then error "Illegal timeout value: $timeout"; fi
            shift
            ;;

        -*)
            error "Unrecognized option: $1"
            ;;

        *)
            break
            ;;
        esac
    done

    if [[ $timeout -ne 0  &&  ! "$default" ]]; then
        error "Non-zero timeout requires a default answer"
    fi

    if [[ ! "$*" ]]; then error "Missing question"; fi

    while [[ $ok -eq 0 ]]
    do
        if [[ $timeout -ne 0 ]]; then
            if ! read -t $timeout -p "$*" ans; then
                ans=$default
            else
                # Turn off timeout if answer entered.
                timeout=0
                if [[ ! "$ans" ]]; then ans=$default; fi
            fi
        else
            read -p "$*" ans
            if [[ ! "$ans" ]]; then
                ans=$default
            else
                ans=$(tr '[:upper:]' '[:lower:]' <<<$ans)
            fi 
        fi

        if [[ "$ans" == 'y'  ||  "$ans" == 'yes'  ||  "$ans" == 'n'  ||  "$ans" == 'no' ]]; then
            ok=1
        fi

        if [[ $ok -eq 0 ]]; then warning "Valid answers are: yes y no n"; fi
    done
    [[ "$ans" = "y" || "$ans" == "yes" ]]
}


if [ -d "${LNDESTDIR}" ]; then
    echo "$LNDESTDIR exists"
    if yesno --timeout 5 --default no "Do You Want to DELETE the exists (timeout 5, default no) ? "; then
        rm -R "$LNDESTDIR"
        echo "$LNDESTDIR deleted."
        rm -R "$DESTDIR"
        echo "$DESTDIR deleted."
    fi
fi
if [ -d "${LNDESTDIR}" ]; then
    echo "$LNDESTDIR exists"
else
    echo "$LNDESTDIR NOT exists"

    mkdir ${DESTDIR}
    mkdir ${DESTDIR}/${DESTDIRNAME}
    cp -r ${SRCDIR}/Annotations ${DESTDIR}/${DESTDIRNAME}/
    cp -r ${SRCDIR}/ImageSets ${DESTDIR}/${DESTDIRNAME}/
    mkdir ${DESTDIR}/${DESTDIRNAME}/JPEGImages

    TRAINTXT=${DESTDIR}/${DESTDIRNAME}/ImageSets/Main/train.txt

    # http://stackoverflow.com/questions/4165135/how-to-use-while-read-bash-to-read-the-last-line-in-a-file-if-there-s-no-new
    while IFS= read -r line || [[ -n "$line" ]]; do
      cp ${SRCDIR}/JPEGImages/${line}.jpg ${DESTDIR}/${DESTDIRNAME}/JPEGImages/
    done <${TRAINTXT}

    ln -s ${DESTDIR} ${LNDESTDIR}

fi

if [ -d "${PYFASTRCNNCACHE}" ]; then
    if yesno --timeout 5 --default no "Do You Want to DELETE ${PYFASTRCNNCACHE} (timeout 5, default no) ? "; then
        rm -R "$PYFASTRCNNCACHE"
        echo "$PYFASTRCNNCACHE deleted."
    fi
fi


