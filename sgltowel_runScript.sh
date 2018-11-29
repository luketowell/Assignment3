usage(){
        echo $0 numOfRepeats numOMPThreads numGPUThreadsPC arrayLength 
}

if [[ $# -ne 4 ]]; then
        usage
        exit -1
fi

EXE=./sgltowel_Code.exe
OMPTHREADS=$1
GPUTHREADS=$2
REPEATS=$3
ARRAY_LENGTH=$4
FILE=CUDAfor${ARRAY_LENGTH}on${GPUTHREADS}by${REPEATS}.txt

#Work out number of blocks
let GPUBLOCKS=($4+$2-1)/$2

echo best of $REPEATS for executable $EXE

        export OMP_NUM_THREADS=$THREADS
        echo running ${EXE} with an array length of ${ARRAY_LENGTH}
        echo running on ${GPUTHREADS} threads across ${GPUBLOCKS} blocks.
        #for k in `seq 1 $REPEATS`; do
                
        #done  > $FILE
        echo outputting data to ${FILE}

echo Finished