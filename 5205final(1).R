library(readr)
library(dplyr)
library(tidyr)
library(keras)
library(tidyverse)

# Load and inspect the dataset
data <- read_csv("C:/Users/tuqqa/Downloads/NYPD_Complaint_Data_Historic_20240419.csv")

data1 <- subset(data, select = -c(RPT_DT,LOC_OF_OCCUR_DESC,PARKS_NM,HADEVELOPT,HOUSING_PSA,SUSP_AGE_GROUP,SUSP_RACE,SUSP_SEX,STATION_NAME,VIC_AGE_GROUP,VIC_RACE,CMPLNT_NUM, CMPLNT_TO_DT, CMPLNT_TO_TM, ADDR_PCT_CD, KY_CD, PD_CD, JURISDICTION_CODE, X_COORD_CD, Y_COORD_CD, TRANSIT_DISTRICT, Latitude, Longitude,JURIS_DESC) )
contains_unwanted <- function(x) {
  grepl("(null)|UNKNOWN", x, ignore.case = TRUE)
}
crimedata <- data1 %>%
  filter_all(all_vars(!contains_unwanted(.)))

crimedata <- drop_na(crimedata)
data <- crimedata
# Convert character columns to factors where appropriate
data$CRM_ATPT_CPTD_CD <- as.factor(data$CRM_ATPT_CPTD_CD)
data$PREM_TYP_DESC <- as.factor(data$PREM_TYP_DESC)
data$PATROL_BORO <- as.factor(data$PATROL_BORO)
data$VIC_SEX <- as.factor(data$VIC_SEX)

# Remove the Lat_Lon column as it requires special handling
data$Lat_Lon <- NULL

# One-hot encoding using model.matrix()
data <- data %>%
  mutate(across(where(is.factor), as.character)) %>%
  mutate(across(where(is.character), as.factor))

if ("CMPLNT_FR_DT" %in% names(data)) {
  # Get the top 10 most frequent levels
    frequent_levels <- names(sort(table(data$CMPLNT_FR_DT), decreasing = TRUE)[1:10])
    # Factor with reduced levels
      data$CMPLNT_FR_DT <- factor(ifelse(data$CMPLNT_FR_DT %in% frequent_levels, data$CMPLNT_FR_DT , 'Other'))
      } else {    
        stop("Column CMPLNT_FR_DT does not exist in the dataset")}
if ("PD_DESC" %in% names(data)) {
  # Get the top 10 most frequent levels
  frequent_levels <- names(sort(table(data$PD_DESC), decreasing = TRUE)[1:10])
  # Factor with reduced levels
  data$PD_DESC <- factor(ifelse(data$PD_DESC %in% frequent_levels, data$CMPLNT_FR_DT , 'Other'))
} else {    
  stop("Column PD_DESC does not exist in the dataset")}

response <- model.matrix(~ LAW_CAT_CD - 1, data = data)
features <- model.matrix(~ . - 1 - LAW_CAT_CD, data = data)

# Split the data
set.seed(617)
indices <- sample(1:nrow(data), 0.8 * nrow(data))
train_features <- features[indices, ]
test_features <- features[-indices, ]
train_labels <- response[indices, ]
test_labels <- response[-indices, ]

# Define the neural network
input_shape <- ncol(train_features) # Number of features
model <- keras_model_sequential()
model$add(layer_dense(units = 64, input_shape = c(ncol(train_features)), activation = 'relu'))
model$add(layer_dense(units = 3, activation = 'softmax'))  # Changed from 10 to 3 to match the number of classes

# Compile the model
model$compile(
  loss = 'categorical_crossentropy',
  optimizer = 'adam',
  metrics = list('accuracy')
)

# Train the model
history <- model$fit(
  train_features, train_labels,
  epochs = as.integer(10),
  batch_size = as.integer(32),
  validation_split = as.double(0.2)
)

# Evaluate the model
score <- model$evaluate(test_features, test_labels)
cat("Test loss:", score[1], "Test accuracy:", score[2], "\n")


# Extract metrics from the history object
epochs <- seq_len(length(history$history$loss))
train_loss <- history$history$oss
val_loss <- history$history$val_loss
train_accuracy <- history$history$acc
val_accuracy <- history$history$val_acc

# Create a data frame for plotting
history_df <- data.frame(
  epoch = epochs,
  train_loss = train_loss,
  val_loss = val_loss,
  train_accuracy = train_accuracy,
  val_accuracy = val_accuracy
)

# Plotting training and validation loss
ggplot(history_df, aes(x = epoch)) +
  geom_line(aes(y = train_loss, colour = "Training Loss")) +
  geom_line(aes(y = val_loss, colour = "Validation Loss")) +
  labs(title = "Training and Validation Loss per Epoch",
       x = "Epoch",
       y = "Loss") +
  scale_colour_manual("", 
                      breaks = c("Training Loss", "Validation Loss"),
                      values = c("blue", "red"))

# Plotting training and validation accuracy
ggplot(history_df, aes(x = epoch)) +
  geom_line(aes(y = train_accuracy, colour = "Training Accuracy")) +
  geom_line(aes(y = val_accuracy, colour = "Validation Accuracy")) +
  labs(title = "Training and Validation Accuracy per Epoch",
       x = "Epoch",
       y = "Accuracy") +
  scale_colour_manual("", 
                      breaks = c("Training Accuracy", "Validation Accuracy"),
                      values = c("green", "purple"))


library(pROC)

predicted_probs <- model$predict(test_features)

# Preparing an empty list to store ROC curves
roc_curves <- list()
auc_values <- numeric(ncol(test_labels))

# Calculate ROC and AUC for each class
for (i in 1:ncol(test_labels)) {
  roc_curves[[i]] <- roc(test_labels[, i], predicted_probs[, i])
  auc_values[i] <- auc(roc_curves[[i]])
}

# Print AUC values for each class
print(auc_values)

plot(roc_curves[[1]], col = 1, main = "ROC Curves for Each Class")
for (i in 2:length(roc_curves)) {
  plot(roc_curves[[i]], add = TRUE, col = i)
}
legend("bottomright", legend = paste("Class", 1:length(roc_curves)), col = 1:length(roc_curves), lty = 1)

##actual_classes
actual_classes <- apply(test_labels, 1, which.max) - 1

library(ggplot2)

# Extract the table of results from the confusion matrix
matrix_table <- as.data.frame(confusionMatrix$table)

# Plot using ggplot2
ggplot(data = matrix_table, aes(x = Reference, y = Prediction, fill = Freq)) + 
  geom_tile() +
  geom_text(aes(label = Freq), vjust = 1) + 
  scale_fill_gradient(low = "blue", high = "red") +
  labs(x = 'Actual Category', y = 'Predicted Category') +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))