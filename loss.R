library(readr)
library(reshape2)
library(ggplot2)

base_Dir <- 'D:/WaveNet.LJFV.Results/Inference/'
loss.Data <- read_delim(
  sprintf('%slog.txt', base_Dir),
  "\t",
  col_names = FALSE,
  escape_double = FALSE,
  locale = locale(encoding = "UTF-8"),
  trim_ws = TRUE
)[c(2,4)]
colnames(loss.Data) <- c('Step', 'Loss')

# loss.Data <- subset(loss.Data, loss.Data$Step > 15000)

plot <- ggplot(data= loss.Data, aes(x=Step, y=Loss)) +
  geom_line() +
  stat_smooth(color='blue', fill='grey')

ggsave(
  filename = sprintf('%sloss.png', base_Dir),
  plot = plot,
  device = "png", width = 50, height = 10, units = "cm", dpi = 300
  )