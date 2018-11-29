usage(){
        echo $0 numOfRepeats numOMPThreads numGPUThreadsPC arrayLength 
}

if [[ $# -ne 4 ]]; then
        usage
        exit -1
fi

EXE=sgltowel_code.exe
OMPTHREADS=$2
GPUTHREADS=$3
REPEATS=$1
DATAPOINTS=$4
FILE=CUDAfor${DATAPOINTS}on${GPUTHREADS}by${REPEATS}.txt

#Work out number of blocks
let GPUBLOCKS=($4+$3-1)/$3

echo best of $REPEATS for executable $EXE

        export OMP_NUM_THREADS=$OMPTHREADS
        echo running ${EXE} with an array length of ${DATAPOINTS}
        echo running on ${GPUTHREADS} threads across ${GPUBLOCKS} blocks.
        for k in `seq 1 $REPEATS`; do
            ./${EXE} ${DATAPOINTS} ${GPUTHREADS}            
        done  > $FILE
        echo outputting data to ${FILE}

echo Finished