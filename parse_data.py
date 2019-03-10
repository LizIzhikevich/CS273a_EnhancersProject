from Bio import SeqIO
import pickle
import sys

# error exit
def error_exit():
	print('parse_data.py <fafile> <outfile>')
	sys.exit(2)

# return true if positive, false if negative
def is_enhancer( seq_record ):
	identify = seq_record.description.split('|')[3]
	if 'positive' in identify:
		return True
	else:
		return False
#return the sequences
def extract_sequences( seq_record ):
	my_seq = str(seq_record.seq)
	return my_seq

#true if human, false if mouse
def extract_human( seq_record ):
	identify = seq_record.description.split('|')[0]
	if 'human' in identify:
		return True
	else:
		return False

#return a list of all the extra features 
#assumes record is positive
def extract_enhancer_features( seq_record ):
	if not is_enhancer( seq_record ):
		return []
	features = seq_record.description.split('|')[4:]	
	returning_features = []

	# forebrain, midbrain, hindbrain, limb, and heart
	for item in features:
		if 'forebrain' in item:
			returning_features.append( 'forebrain' )
		if 'midbrain' in item:
			returning_features.append( 'midbrain' )
		if 'hindbrain' in item:
			returning_features.append( 'hindbrain' ) 
		if 'limb' in item:
			returning_features.append( 'limb' )
		if 'heart' in item:
			returning_features.append( 'heart' )

	return returning_features

# build a one_hot_encoding dict for when we extract enhancer features (limb, forebrain,etc) 
#def build_feature_dict( file ):

#for seq_record in SeqIO.parse("data", "fasta"):
def main(args):
	if len(args) < 2:
		error_exit()
	fafile = args[0]
	outfile = args[1]

	#build_feature_dict( file )
	data = []
	for seq_record in SeqIO.parse(fafile, "fasta"):
		data.append( [extract_sequences( seq_record ), 
				is_enhancer( seq_record ), 
				extract_enhancer_features( seq_record )] )
	print('Number of samples: %d\n' % len(data))

	file = open(outfile, 'wb')
	pickle.dump(data, file)
	file.close()

if __name__ == "__main__":
	main(sys.argv[1:])
