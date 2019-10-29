#example of PowerShell script for training the model w/ different parameters
conda activate tf-gpu
$epochs = 70
$subdir = "paper"
$classes = (7)
foreach ($class_n in $classes) {
    $cmd = "python main.py --epochs=$epochs --subdir=$($class_n)_$($subdir) --classes=$class_n";
    $cmd;
    Invoke-Expression -Command $cmd; 
}