import sys
import time
import argparse
from genetic import GeneticAlgorithm
from gradient import GradientEstimation


parser = argparse.ArgumentParser(description='Run the SMACK attack.')

# Add the arguments
parser.add_argument('--audio',
                    type=str,
                    required=True,
                    help='The original speech audio path')

parser.add_argument('--model',
                    type=str,
                    required=True,
                    help='The target model can be "googleASR" or "iflytekASR" or "gmmSV" or "ivectorSV"')

parser.add_argument('--content',
                    type=str,
                    required=True,
                    help='The reference speech content in the audio.')

parser.add_argument('--target',
                    type=str,
                    required=True,
                    help='The target sentence for ASR systems or identity number for SV systems.')

# Parse the arguments
args = parser.parse_args()


reference_audio = args.audio
reference_text = args.content
# target_model can be 'googleASR' or 'iflytekASR' or 'gmmSV' or 'ivectorSV'
target_model = args.model
target = args.target
population_size = 20
genetic_iterations = 10
gradient_iterations = 5

# Record the start time
start_time = time.time()

ga = GeneticAlgorithm(reference_audio, reference_text, target_model, target, population_size)

# Run the Genetic Algorithm
if 'ASR' in target_model:
    fittest_individual = ga.run(genetic_iterations)
    threshold_range = None
    
elif 'SV' in target_model:
    fittest_individual, threshold_range = ga.run(genetic_iterations)
    
elif 'CSI' in target_model:
    fittest_individual = ga.run(genetic_iterations)
    threshold_range = None
    
elif 'OSI' in target_model:
    fittest_individual, threshold_range = ga.run(genetic_iterations)

print("The adapted genetic algorithm finished. Now launching the gradient estimation. \n")

# Initialize the GradientEstimation
gradient_estimator = GradientEstimation(reference_audio, reference_text, target_model, target, threshold_range, sigma=0.1, learning_rate=0.01, K=20)

# Run the Gradient Estimation
p_refined = gradient_estimator.refine_prosody_vector(fittest_individual, gradient_iterations)

# Record the end time
end_time = time.time()

# Calculate and display the elapsed time
elapsed_time = end_time - start_time
elapsed_hours = int(elapsed_time // 3600)
elapsed_minutes = int((elapsed_time % 3600) // 60)
elapsed_seconds = elapsed_time % 60

print(f"The adapted gradient estimation finished. Time elapsed: {elapsed_hours} hours, {elapsed_minutes} minutes, and {elapsed_seconds:.2f} seconds")